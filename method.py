import cv2
import numpy as np

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    전역 히스토그램 평활화(Histogram Equalization)를 적용한다.
    그레이스케일 이미지는 단일 채널로, 컬러 이미지는 채널별로 적용한다.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        eq_img = cv2.merge((b_eq, g_eq, r_eq))
    else:
        eq_img = cv2.equalizeHist(img)
    return eq_img

def clahe_enhancement(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    CLAHE(Contrast Limited Adaptive Histogram Equalization)를 적용한다.
    clip_limit은 대비 제한 설정값, tile_grid_size는 지역 히스토그램 적용 구역 크기다.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        b_clahe = clahe.apply(b)
        g_clahe = clahe.apply(g)
        r_clahe = clahe.apply(r)
        clahe_img = cv2.merge((b_clahe, g_clahe, r_clahe))
    else:
        clahe_img = clahe.apply(img)
    return clahe_img

def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    감마 보정(Gamma Correction)을 적용한다.
    gamma > 1.0 이면 어두운 영역이 상대적으로 더 밝아진다.
    """
    img_float = img.astype(np.float32) / 255.0
    corrected = np.power(img_float, gamma)
    corrected_img = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    return corrected_img

def single_scale_retinex(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Single Scale Retinex (SSR) 기법을 적용한다.
    SSR = log(I) - log(가우시안블러(I))
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        b_retinex = _ssr_channel(b, sigma)
        g_retinex = _ssr_channel(g, sigma)
        r_retinex = _ssr_channel(r, sigma)
        retinex_img = cv2.merge((b_retinex, g_retinex, r_retinex))
    else:
        retinex_img = _ssr_channel(img, sigma)
    return retinex_img

def _ssr_channel(channel: np.ndarray, sigma: float) -> np.ndarray:
    channel_float = channel.astype(np.float32) + 1e-6
    log_channel = np.log(channel_float)
    blur = cv2.GaussianBlur(channel_float, ksize=(0,0), sigmaX=sigma, sigmaY=sigma) + 1e-6
    log_blur = np.log(blur)
    retinex = log_channel - log_blur
    retinex = cv2.normalize(retinex, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return retinex

def shadow_highlight(img: np.ndarray, shadow: float = 0.5, highlight: float = 0.5) -> np.ndarray:
    """
    간단한 Shadow/Highlight 보정 기법을 흉내낸 예시.
    shadow: 어두운 영역 밝기 보정 (0 ~ 1 사이 값 권장)
    highlight: 밝은 영역 억제 또는 보정 (0 ~ 1 사이 값 권장)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray_float = gray.astype(np.float32) / 255.0
    
    shadow_mask = (1.0 - gray_float) ** 2
    highlight_mask = gray_float ** 2

    img_float = img.astype(np.float32) / 255.0

    # shadow_mask가 클수록(어두울수록) shadow 인자만큼 밝기를 끌어올림
    # highlight_mask가 클수록(밝을수록) highlight 인자만큼 값을 억제
    # 간단한 공식 예시:  
    #   shadowed = original + shadow_mask * shadow * (1 - original)
    #   highlighted = shadowed - highlight_mask * highlight * shadowed
    # 등으로 구현할 수 있지만, 다양한 변형이 가능하다.
    
    # 여기서는 단순 가중 합으로 처리
    shadowed = img_float + (shadow_mask[..., None] * shadow * (1 - img_float))
    corrected = shadowed - (highlight_mask[..., None] * highlight * shadowed)
    
    result = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    return result

def simple_tone_mapping(img: np.ndarray, gamma: float = 2.2, intensity: float = 0.4) -> np.ndarray:
    """
    간단한 톤 매핑 기법. (OpenCV의 TonemapDrago, TonemapReinhard 등도 활용 가능)
    intensity: 톤 매핑 적용 강도 (0~1 사이 정도로 가정)
    """
    # 1) 감마 보정으로 일차적인 조정
    corrected = gamma_correction(img, gamma)

    # 2) 이미지 전체의 평균 밝기를 특정 범위에 맞추는 예시
    # intensity가 클수록 바뀐 값을 더 많이 반영
    gray_corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
    avg_brightness = np.mean(gray_corrected)
    target_brightness = 128.0
    adjust_factor = (target_brightness / (avg_brightness + 1e-6)) - 1.0

    # 수정 적용
    float_img = corrected.astype(np.float32)
    mapped = float_img + adjust_factor * intensity * float_img
    mapped = np.clip(mapped, 0, 255).astype(np.uint8)
    return mapped
