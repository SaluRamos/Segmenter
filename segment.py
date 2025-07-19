import os
import cv2
import numpy as np
import torch
from PIL import Image
import colorsys
import shutil
from enum import Enum
import time

# --------------------------------------------- GENERAL UTILS ---------------------------------------------

def generate_distinct_colors(n, s=0.8, v=0.95):
    colors = []
    if SEGMENTATION_COLOR_TYPE == SegmentationColorType.COLORED:
        hues = np.linspace(0, 1, n, endpoint=False)
        for h in hues:
            rgb = colorsys.hsv_to_rgb(h, s, v)  # valores de 0-1
            rgb = tuple(int(c * 255) for c in rgb)  # converte para 0-255
            colors.append(rgb)
    elif SEGMENTATION_COLOR_TYPE == SegmentationColorType.GRAYSCALE:
        for i in range(0, n):
            colors.append((i,) * 3)
    return colors

class ChosenModel(Enum):
    SAM = 1
    SAM2 = 2
    SEMANTIC_SAM = 3
    MASK2FORMER = 4

DINOV2_PATCH_SIZE = 14

class ConstSemanticAssistant(Enum):
    NONE = 0 #usado para casos em que o modelo possui category_id (ex: Mask2Former)
    CLIP = 1
    DINO_V2 = 2

class SemanticAssistantCropStyle(Enum):
    BLACK = 1
    BLUR = 2
    MEAN = 3

class SegmentationColorType(Enum):
    COLORED = 1
    GRAYSCALE = 2

def crop_image_region(image:np.ndarray, mask:np.ndarray, bbox:list, mask_path:str):
    region = image.copy()
    x, y, w, h = [int(val) for val in bbox]
    region = region[y:y + h, x:x + w]
    mask = mask[y:y + h, x:x + w]
    if SEMANTIC_ASSISTANT_CROP_STYLE == SemanticAssistantCropStyle.BLACK:
        region[~mask] = 0
    elif SEMANTIC_ASSISTANT_CROP_STYLE == SemanticAssistantCropStyle.BLUR:
        kernel_blur_size = 101
        if kernel_blur_size < 0:
            raise Exception("kernel blur size must be bigger than 0")
        if kernel_blur_size % 2 == 0:
            raise Exception("kernel blur size must be odd")
        blurred = cv2.GaussianBlur(image, (kernel_blur_size, kernel_blur_size), 0)
        blurred = blurred[y:y + h, x:x + w]
        region = np.where(mask[..., None], region, blurred)
    elif SEMANTIC_ASSISTANT_CROP_STYLE == SemanticAssistantCropStyle.MEAN:
        mean_color = region[mask].mean(axis=0)
        region[~mask] = mean_color.astype(np.uint8)

    if SAVE_MASKS:
        cv2.imwrite(mask_path, cv2.cvtColor(region, cv2.COLOR_RGB2BGR))

    return region

def generate_seg_image_with_sam(mask_generator, src_folder, dst_folder, file):
    src_path = os.path.join(src_folder, file)
    dst_path_ref = os.path.join(dst_folder, os.path.splitext(file)[0] + "_ref.png")
    dst_path = os.path.join(dst_folder, file)
    dst_path_ghost = os.path.join(dst_folder, os.path.splitext(file)[0] + "_ghost.png")

    #observação: a original_image sempre é RGB!
    if CHOSEN_MODEL == ChosenModel.SAM:
        original_image = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(original_image)
    elif CHOSEN_MODEL == ChosenModel.SEMANTIC_SAM:
        original_image, input_image = prepare_image(image_pth=src_path)
        masks = mask_generator.generate(input_image)
    elif CHOSEN_MODEL == ChosenModel.SAM2:
        original_image = Image.open(src_path)
        original_image = np.array(original_image.convert("RGB"))
        masks = mask_generator.generate(original_image)

    # masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    #o ideal é unir mascaras com conteudo semelhante (panoptica para semantica) antes de passar para o CLIP, afim de entregar mais detalhes/contexto
    #entretanto isso não é possivel pois o SAM não faz segmentação inteligente, ele apenas segmenta com base no objeto.

    mask_colored = np.zeros_like(original_image)
    mask_colored_copy = None

    if SEMANTIC_ASSISTANT == ConstSemanticAssistant.CLIP:
        labeled_regions = []
        candidates = classify_image_with_clip(original_image, 0.20, PROMPTS)
        for mask_idx, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation']
            mask_path = "data/dataset_labels/" + os.path.splitext(file)[0] + "_mask_" + str(mask_idx) + ".png"
            label = classify_region_with_clip(original_image, mask, mask_dict['bbox'], mask_path, candidates)
            labeled_regions.append((label, mask))
            color = COLOR_MAP.get(label, (255, 255, 255))
            mask_colored[mask] = color
        mask_colored_copy = mask_colored.copy() #fazer cópia do mask_colored antes de aplciar textos
        for label, mask in labeled_regions:
            mask_colored = draw_label_on_mask(mask_colored, mask, label, (255, 255, 255))
    elif SEMANTIC_ASSISTANT == ConstSemanticAssistant.DINO_V2:
        for mask_idx, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation']
            mask_path = "data/dataset_labels/" + os.path.splitext(file)[0] + "_mask_" + str(mask_idx) + ".png"
            id = extract_features_with_dinov2(original_image, mask, mask_dict['bbox'], mask_path)
            if id == None:
                continue
            color = COLOR_MAP[id]
            mask_colored[mask] = color
        mask_colored_copy = mask_colored.copy()


    if SAVE_SEGMENTATION_WITH_ORIGINAL_GHOST:
        mask_colored_copy = cv2.addWeighted(original_image, GHOST_ALPHA, mask_colored_copy, 1 - GHOST_ALPHA, 0)
        Image.fromarray(mask_colored_copy).save(dst_path_ghost)

    final_img = Image.fromarray(mask_colored)
    final_img.save(dst_path)

    if COPY_REFERENCE_TO_OUTPUT_FOLDER:
        shutil.copy(src_path, dst_path_ref)

# --------------------------------------------- CLIP UTILS ---------------------------------------------

def classify_image_with_clip(image:np.ndarray, min_treshold:float, prompts:list[str]):
    region = image.copy()
    region_pil = Image.fromarray(region)
    region_clip = preprocess_clip(region_pil).unsqueeze(0).to(DEVICE)
    text_tokens = tokenizer(prompts).to(DEVICE)

    image_feat = model_clip.encode_image(region_clip)
    text_feats = model_clip.encode_text(text_tokens)
    image_feat /= image_feat.norm(dim=-1, keepdim=True)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    candidates = []
    similarity = (image_feat @ text_feats.T).squeeze(0).cpu().numpy()
    pairs = list(zip(prompts, similarity))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    print("Probabilidades:")
    for prompt, prob in pairs_sorted:
        print(f"{prompt:24s}: {prob:.4f}")
        if prob > min_treshold:
            candidates.append(prompt)
    print(f"candidates: {candidates}")
    return candidates

def classify_region_with_clip(image:np.ndarray, mask:np.ndarray, bbox:list, mask_path:str, prompts:list[str]):
    region = crop_image_region(image, mask, bbox, mask_path)

    region_pil = Image.fromarray(region)
    region_clip = preprocess_clip(region_pil).unsqueeze(0).to(DEVICE)
    text_tokens = tokenizer(prompts).to(DEVICE)

    image_feat = model_clip.encode_image(region_clip)
    text_feats = model_clip.encode_text(text_tokens)
    image_feat /= image_feat.norm(dim=-1, keepdim=True)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    similarity = image_feat @ text_feats.T
    best_idx = similarity.argmax().item()
    return prompts[best_idx]

def draw_label_on_mask(image, mask, label, color, font_scale=0.6, thickness=1):
    # Encontrar o centro da máscara
    y_indices, x_indices = mask.nonzero()
    if len(x_indices) == 0 or len(y_indices) == 0:
        return image
    center_x = int(x_indices.mean())
    center_y = int(y_indices.mean())
    # Tamanho da imagem
    img_h, img_w = image.shape[:2]
    # Fonte e tamanho do texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_width, text_height), _) = cv2.getTextSize(label, font, font_scale, thickness)
    # Ajuste de posição para manter o texto dentro da imagem
    text_x = max(0, min(center_x - text_width // 2, img_w - text_width))
    text_y = max(text_height, min(center_y + text_height // 2, img_h - 1))
    # Contorno do texto (borda preta)
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),  # borda
        thickness + 2,
        cv2.LINE_AA,
    )
    # Texto principal
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        font,
        font_scale,
        tuple(int(c) for c in color),
        thickness,
        cv2.LINE_AA,
    )
    return image

# --------------------------------------------- DINOV2 UTILS ---------------------------------------------

def load_crop_features_and_index():
    CROP_FEATURE_FILES = {
        'real': ('data/crop_real.npz', 'data/crop_real.csv'),
        'game': ('data/crop_game.npz', 'data/crop_game.csv'),
    }
    all_features = {}
    path_to_feature = {}
    for domain, (npz_path, csv_path) in CROP_FEATURE_FILES.items():
        #domain can be 'real' or 'game'
        if not os.path.exists(npz_path):
            raise Exception(f"file {npz_path} doesnt exist")
        if not os.path.exists(csv_path):
            raise Exception(f"file {csv_path} doesnt exist")
        features = np.load(npz_path)['crops']
        with open(csv_path, 'r') as f:
            next(f) #pula header
            for idx, line in enumerate(f):
                _, path, *_ = line.strip().split(',')
                filename = path.split('\\')[-1]
                path_to_feature.setdefault(filename, []).append((domain, idx))
                # print(f"mapping {filename} -> {domain} , {idx}")
        all_features[domain] = features
    return all_features, path_to_feature

def find_cluster_centroids(sample_size:int):
    """Seleciona features amostradas dos arquivos crop_*.npz e executa K-Means."""
    print(f"Selecting {sample_size} vectors from ALL_FEATURES for clustering...")
    sample_size = min(sample_size, sum([v.shape[0] for v in ALL_FEATURES.values()]))

    all_vectors = []
    for domain, features in ALL_FEATURES.items():
        all_vectors.append(features)
    all_vectors = np.vstack(all_vectors)
    
    total = all_vectors.shape[0]
    if total > sample_size:
        indices = np.random.choice(total, size=sample_size, replace=False)
        sample = all_vectors[indices]
    else:
        sample = all_vectors
    
    print(f"Executing K-Means to find {AMOUNT_CLUSTERS} centroids with {sample.shape[0]} vectors...")
    d = sample.shape[1]
    use_gpu = True if torch.cuda.is_available() else False
    kmeans = faiss.Kmeans(d, AMOUNT_CLUSTERS, niter=20, verbose=True, gpu=use_gpu)
    kmeans.train(sample.astype(np.float32))
    print("K-Means finished")

    centroids = kmeans.centroids

    if NORMALIZE_LATENT_VECTORS:
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-6

    return centroids

def extract_features_with_dinov2(image:np.ndarray, mask:np.ndarray, bbox:list, mask_path:str):
    region = image.copy()
    x, y, w, h = bbox
    #reduzir w e h para o primeiro multiplo de 14
    w = (w // DINOV2_PATCH_SIZE) * DINOV2_PATCH_SIZE
    h = (h // DINOV2_PATCH_SIZE) * DINOV2_PATCH_SIZE
    if h == 0 or w == 0:
        print(f"Warning: Bounding box with dimensions [{w}, {h}] is too small to be processed. Skipping.")
        return None

    bbox = [x, y, w, h]
    region = crop_image_region(image, mask, bbox, mask_path)

    pil_img = Image.fromarray(region)
    input_tensor = DINOV2_NORMALIZE(pil_img).unsqueeze(0).to(DEVICE)
    output = DINOV2_MODEL(input_tensor)

    if len(output.shape) == 3: # (B, N, D)
        feature_vector = output[:, 0] # CLS token
    else:
        feature_vector = output # caso já seja vetorial (nem sempre acontece)

    feature_vector = feature_vector.cpu().numpy().astype(np.float32).squeeze()

    if NORMALIZE_LATENT_VECTORS:
        feature_vector /= np.linalg.norm(feature_vector) + 1e-6

    print(f"{mask_path}: ")
    print(feature_vector)
    similarities = np.dot(CENTROIDS, feature_vector) # (N_clusters, 1)
    closest_cluster_idx = int(np.argmax(similarities))

    return closest_cluster_idx

# --------------------------------------------- MODELS ---------------------------------------------

def apply_sam(files, src_folder, dst_folder):
    print("INITIALIZING SAM...")
    sam_checkpoint_h = "models/sam_vit_h_4b8939.pth" # 2.5 GB
    sam_checkpoint_l = "models/sam_vit_l_0b3195.pth" # 1.2 GB
    sam_checkpoint_b = "models/sam_vit_b_01ec64.pth" # 366 MB
    model_type = "vit_l" # Escolha entre: vit_h, vit_l, vit_b

    sam = None
    if model_type == "vit_b":
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_b)
    elif model_type == "vit_l":
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_l)
    elif model_type == "vit_h":
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_h)
    else:
        raise Exception(f"model type {model_type} doesnt exist for SAM ")

    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32, #Número de pontos de amostragem por lado da imagem
        pred_iou_thresh=0.88, #Threshold de qualidade para manter máscara
        stability_score_thresh=0.95, #Score de estabilidade da máscara (ruído vs confiança)
        crop_n_layers=0, 
        #0 = desliga recortes — só roda SAM na imagem inteira.
        #1 = faz uma divisão da imagem em recortes, detecta objetos por crop.
        #2 = faz duas camadas de recortes (mais refinado, mais lento).
        crop_n_points_downscale_factor=2, #Menor valor = mais pontos por crop = mais preciso/lento, default 2
        min_mask_region_area=MIN_AREA_SIZE  #Remove máscaras menores que essa área (em pixels)
    )

    for index, file in enumerate(files):
        print(f"[SAM] Processando: {index + 1}/{len(files)} - {file}")
        generate_seg_image_with_sam(mask_generator, src_folder, dst_folder, file)

def apply_sam2(files, src_folder, dst_folder):
    print("INITIALIZING SAM2...")
    checkpoint = "models/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1/sam2.1_hiera_l.yaml"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base="1.3", config_dir="/mnt/e/epe/sam2/sam2/configs")

    sam2 = build_sam2(model_cfg, checkpoint, device=DEVICE, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=16,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        stability_score_offset=0.9,
        crop_n_layers=0,
        box_nms_thresh=0.5,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=MIN_AREA_SIZE,
        use_m2m=False
    )

    for index, file in enumerate(files):
        print(f"[SAM2] Processando: {index + 1}/{len(files)} - {file}")
        generate_seg_image_with_sam(mask_generator, src_folder, dst_folder, file)

def apply_semantic_sam(files, src_folder, dst_folder):
    print("INITIALIZING SEMANTIC SAM...")
    swin_t_checkpoint = "models/swint_only_sam_many2many.pth" # 207 MB
    swin_l_checkpoint = "models/swinl_only_sam_many2many.pth" # 874 MB
    model_type = "L" # Escolha entre: 'L' ou 'T'

    ckpt = None
    if model_type == "T":
        ckpt = swin_t_checkpoint
    elif model_type == "L":
        ckpt = swin_l_checkpoint
    else:
        raise Exception(f"model type {model_type} doesnt exist for SEMANTIC SAM ")

    cfgs={'T':"Semantic-SAM/configs/semantic_sam_only_sa-1b_swinT.yaml",
          'L':"Semantic-SAM/configs/semantic_sam_only_sa-1b_swinL.yaml"}
    opt = load_opt_from_config_file(cfgs[model_type])
    sam = BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().cuda()
    mask_generator = SemanticSamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        crop_n_layers=0,
        min_mask_region_area=MIN_AREA_SIZE,
        level=[2]
    )
    for index, file in enumerate(files):
        print(f"[SEMANTIC SAM] Processando: {index + 1}/{len(files)} - {file}")
        generate_seg_image_with_sam(mask_generator, src_folder, dst_folder, file)

def apply_mask2former(files, src_folder, dst_folder):
    print("INITIALIZING MASK2FORMER...")

    model = {"config":"Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
             "weights":"models/mask2Former-200queries-panoptic-coco-Swin-L-IN21k.pkl"}
    # model = {"config":"Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml",
    #          "weights":"models/mask2Former-semantic-cityscapes-swin-L-IN21k.pkl"}

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(model["config"])
    cfg.merge_from_list(['MODEL.WEIGHTS', model["weights"]])
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    for index, file in enumerate(files):
        print(f"[MASK2FORMER] Processando: {index + 1}/{len(files)} - {file}")
        src_path = os.path.join(src_folder, file)
        dst_path_ref = os.path.join(dst_folder, os.path.splitext(file)[0] + "_ref.png")
        dst_path = os.path.join(dst_folder, os.path.splitext(file)[0] + ".png")
        dst_path_ghost = os.path.join(dst_folder, os.path.splitext(file)[0] + "_ghost.png")

        img_orig = Image.open(src_path).convert("RGB")
        width, height = img_orig.size
        if width <= 300 or height <= 300:
            print(f"[IGNORADO] {src_path}: imagem muito pequena ({width}x{height})")
            continue
        if width > 1920 or height > 1080:
            print(f"[IGNORADO] {src_path}: imagem muito grande ({width}x{height})")
            continue

        image = read_image(src_path, format="BGR")

        predictions = predictor(image)
        panoptic_seg, segments_info = predictions["panoptic_seg"]

        height, width = panoptic_seg.shape
        mask_img = np.zeros((height, width, 3), dtype=np.uint8)

        for segment in segments_info:
            segment_id = segment["category_id"] #se aqui for só 'id' será panoptica, se for 'category_id' será semantica
            color = COLOR_MAP[segment_id]
            mask = (panoptic_seg == segment["id"]).cpu().numpy()
            mask_img[mask] = color

        if SAVE_SEGMENTATION_WITH_ORIGINAL_GHOST:
            img_orig_np = np.array(img_orig)
            ghost_img = (GHOST_ALPHA * img_orig_np + (1 - GHOST_ALPHA) * mask_img).astype(np.uint8)
            Image.fromarray(ghost_img).save(dst_path_ghost)

        Image.fromarray(mask_img).save(dst_path)

        if COPY_REFERENCE_TO_OUTPUT_FOLDER:
            shutil.copy(src_path, dst_path_ref)

def apply_dinomask(files, src_folder, dst_folder):
    pass

if __name__ == "__main__":

    # print(open_clip.list_pretrained())
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    src_folder = "data/dataset"
    dst_folder = "data/dataset_labels"
    CHOSEN_MODEL = ChosenModel.SAM2
    SEMANTIC_ASSISTANT = ConstSemanticAssistant.DINO_V2
    SEMANTIC_ASSISTANT_CROP_STYLE = SemanticAssistantCropStyle.MEAN
    SEGMENTATION_COLOR_TYPE = SegmentationColorType.COLORED
    SAVE_MASKS = False
    COPY_REFERENCE_TO_OUTPUT_FOLDER = True
    SAVE_SEGMENTATION_WITH_ORIGINAL_GHOST = False
    GHOST_ALPHA = 0.3
    MIN_AREA_SIZE = 40
    NORMALIZE_LATENT_VECTORS = False

    os.makedirs(dst_folder, exist_ok=True)
    # np.set_printoptions(threshold=np.inf)

    if SEMANTIC_ASSISTANT == ConstSemanticAssistant.CLIP:
        print("INITIALIZING CLIP...")
        import open_clip

        PROMPTS = ["underwater open sea", "sky background", "boiling hot lava", "a fire explosion", "sand", "rocks underwater", "metal scrap", "underwater building", "a hand holding a tool", "submarine", "coral", "terrestrial vegetation", "kelps", "fish", "shark", "jellyfish", "lobster", "penguim", "seal", "squid", "a scuba diver", "water bubbles"]
        COLOR_MAP = {}

        for label, color in zip(PROMPTS, generate_distinct_colors(len(PROMPTS))):
            COLOR_MAP[label] = color

        clip_model = {"model_name":"ViT-B-32", "pre_trained_data":"laion2b_s34b_b79k"}
        # clip_model = {"model_name":"ViT-H-14-378-quickgelu", "pre_trained_data":"dfn5b"}
        model_clip, _, preprocess_clip = open_clip.create_model_and_transforms(clip_model["model_name"], pretrained=clip_model["pre_trained_data"])
        tokenizer = open_clip.get_tokenizer(clip_model["model_name"])
        model_clip = model_clip.to(DEVICE, dtype=torch.float32)
    elif SEMANTIC_ASSISTANT == ConstSemanticAssistant.DINO_V2:
        print("INITIALIZING DINOV2...")
        from torchvision import transforms
        import faiss

        AMOUNT_CLUSTERS = 20
        COLOR_MAP = generate_distinct_colors(AMOUNT_CLUSTERS)

        CENTROIDS_FILE = "data/dino_centroids.npy"
        FORCE_RECOMPUTE_CENTROIDS = False
        ALL_FEATURES, PATH_TO_FEATURE = load_crop_features_and_index()
        print(f"Available vectors: {sum([v.shape[0] for v in ALL_FEATURES.values()])}")
        CENTROIDS = None
        if not os.path.exists(CENTROIDS_FILE) or FORCE_RECOMPUTE_CENTROIDS:
            CENTROIDS = find_cluster_centroids(150000)
            np.save(CENTROIDS_FILE, CENTROIDS)
        else:
            print(f"Loading centroids from file: {CENTROIDS_FILE}")
            CENTROIDS = np.load(CENTROIDS_FILE)
        
        print(CENTROIDS)

        DINOV2_MODEL = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        DINOV2_MODEL.eval().to(DEVICE)
        DINOV2_NORMALIZE = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif SEMANTIC_ASSISTANT == ConstSemanticAssistant.NONE:
        # DATASET COCO POSSUI 182 CLASSES DE CLASSIFICAÇÃO
        # DATASET CITYSCAPES POSSUI 19 CLASSES DE CLASSIFICAÇÃO
        # ADE20K CITYSCAPES POSSUI 150 CLASSES DE CLASSIFICAÇÃO
        COLOR_MAP = generate_distinct_colors(182)

    src_files = os.listdir(src_folder)
    src_files = [f for f in src_files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    src_files = src_files[:10]

    with torch.inference_mode():

        if CHOSEN_MODEL == ChosenModel.SAM:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            apply_sam(src_files, src_folder, dst_folder)
        elif CHOSEN_MODEL == ChosenModel.SAM2:
            import hydra
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            apply_sam2(src_files, src_folder, dst_folder)
        elif CHOSEN_MODEL == ChosenModel.SEMANTIC_SAM:
            from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator, build_model
            from semantic_sam.BaseModel import BaseModel
            from utils.arguments import load_opt_from_config_file
            apply_semantic_sam(src_files, src_folder, dst_folder)
        elif CHOSEN_MODEL == ChosenModel.MASK2FORMER:
            import sys
            sys.path.insert(1, os.path.join(sys.path[0], '../Mask2Former/'))
            from detectron2.data import MetadataCatalog
            from detectron2.config import get_cfg
            from detectron2.data.detection_utils import read_image
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.utils.visualizer import ColorMode, Visualizer
            from detectron2.projects.deeplab import add_deeplab_config
            from mask2former import add_maskformer2_config
            apply_mask2former(src_files, src_folder, dst_folder)
        