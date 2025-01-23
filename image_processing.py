import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from method import histogram_equalization, clahe_enhancement, gamma_correction, single_scale_retinex, _ssr_channel, shadow_highlight, simple_tone_mapping
import argparse
import os

def image_processing(args):
    image_path = args.path
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print("이미지를 불러올 수 없습니다.")
        return
    if args.crop is not None:
        origin_width, origin_height = bgr.shape[:2]
        x, y, width, height = args.crop
        width = width if width != 0 else origin_width
        height = height if height != 0 else origin_height
        original_bgr = bgr[x:x+width, y:y+height]
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.35)

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    result_path = os.path.join(args.save_path, "origin.png")
    cv2.imwrite(result_path, original_rgb)
    image_plot = ax.imshow(original_rgb)
    ax.set_title("Image Enhancement")

    # radio_axes에 위치 지정: [left, bottom, width, height]
    radio_ax = plt.axes([0.05, 0.55, 0.15, 0.35], facecolor='lightgray')
    transformation_labels = [
        "HistEq",       # 히스토그램 평활화
        "CLAHE",        # CLAHE
        "Gamma",        # 감마 보정
        "Retinex",      # SSR
        "Shadow/HL",    # Shadow/Highlight
        "ToneMap"       # 간단 톤 매핑
    ]
    radio_buttons = RadioButtons(radio_ax, transformation_labels, active=0)

    clahe_clip_ax = plt.axes([0.2, 0.25, 0.3, 0.03])
    clahe_clip_slider = Slider(
        ax=clahe_clip_ax,
        label='clipLimit',
        valmin=1.0,
        valmax=10.0,
        valinit=2.0,
        valstep=0.1
    )

    clahe_tile_ax = plt.axes([0.2, 0.20, 0.3, 0.03])
    clahe_tile_slider = Slider(
        ax=clahe_tile_ax,
        label='tileGrid',
        valmin=1,
        valmax=32,
        valinit=8,
        valstep=1
    )

    gamma_ax = plt.axes([0.2, 0.15, 0.3, 0.03])
    gamma_slider = Slider(
        ax=gamma_ax,
        label='Gamma',
        valmin=0.1,
        valmax=3.0,
        valinit=1.0,
        valstep=0.01
    )

    retinex_ax = plt.axes([0.2, 0.10, 0.3, 0.03])
    retinex_slider = Slider(
        ax=retinex_ax,
        label='Sigma',
        valmin=1.0,
        valmax=80.0,
        valinit=15.0,
        valstep=1.0
    )

    shadow_ax = plt.axes([0.2, 0.05, 0.3, 0.03])
    shadow_slider = Slider(
        ax=shadow_ax,
        label='Shadow',
        valmin=0.0,
        valmax=1.0,
        valinit=0.5,
        valstep=0.01
    )

    highlight_ax = plt.axes([0.2, 0.00, 0.3, 0.03])
    highlight_slider = Slider(
        ax=highlight_ax,
        label='Highlight',
        valmin=0.0,
        valmax=1.0,
        valinit=0.5,
        valstep=0.01
    )

    tone_ax = plt.axes([0.65, 0.25, 0.3, 0.03])
    tone_slider = Slider(
        ax=tone_ax,
        label='ToneInten',
        valmin=0.0,
        valmax=1.0,
        valinit=0.4,
        valstep=0.01
    )
    
    context = {"result_bgr": original_bgr.copy()}
    def update(val):
        transformation = radio_buttons.value_selected
        bgr_copy = original_bgr.copy()

        if transformation == "HistEq":
            result_bgr = histogram_equalization(bgr_copy)
        elif transformation == "CLAHE":
            clip_val = clahe_clip_slider.val
            tile_val = int(clahe_tile_slider.val)
            result_bgr = clahe_enhancement(bgr_copy, clip_limit=clip_val, tile_grid_size=tile_val)
        elif transformation == "Gamma":
            gamma_val = gamma_slider.val
            result_bgr = gamma_correction(bgr_copy, gamma=gamma_val)
        elif transformation == "Retinex":
            sigma_val = retinex_slider.val
            result_bgr = single_scale_retinex(bgr_copy, sigma=sigma_val)
        elif transformation == "Shadow/HL":
            shadow_val = shadow_slider.val
            highlight_val = highlight_slider.val
            result_bgr = shadow_highlight(bgr_copy, shadow=shadow_val, highlight=highlight_val)
        elif transformation == "ToneMap":
            gamma_val = gamma_slider.val
            tone_val = tone_slider.val
            result_bgr = simple_tone_mapping(bgr_copy, gamma=gamma_val, intensity=tone_val)
        else:
            result_bgr = bgr_copy
        context["result_bgr"] = result_bgr
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        image_plot.set_data(result_rgb)
        fig.canvas.draw_idle()

    def on_key_press(event):
        transformation = radio_buttons.value_selected
        if event.key == 'a':
            if transformation == "HistEq":
                name = f"{transformation}.png"
            elif transformation == "CLAHE":
                clip_val = clahe_clip_slider.val
                tile_val = int(clahe_tile_slider.val)
                name = f"{transformation}_{clip_val}_{tile_val}.png"
            elif transformation == "Gamma":
                gamma_val = gamma_slider.val
                name = f"{transformation}_{gamma_val}.png"
            elif transformation == "Retinex":
                sigma_val = retinex_slider.val
                name = f"{transformation}_{sigma_val}.png"
            elif transformation == "Shadow/HL":
                shadow_val = shadow_slider.val
                highlight_val = highlight_slider.val
                name = f"{transformation}_{shadow_val}_{highlight_val}.png"
            elif transformation == "ToneMap":
                gamma_val = gamma_slider.val
                tone_val = tone_slider.val
                name = f"{transformation}_{gamma_val}_{tone_val}.png"
            else:
                name = "origin.png"
            result_path = os.path.join(args.save_path, name)
            cv2.imwrite(result_path, context["result_bgr"])
            print(f"save result : {result_path}")

    radio_buttons.on_clicked(update)
    clahe_clip_slider.on_changed(update)
    clahe_tile_slider.on_changed(update)
    gamma_slider.on_changed(update)
    retinex_slider.on_changed(update)
    shadow_slider.on_changed(update)
    highlight_slider.on_changed(update)
    tone_slider.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to dataset')
    parser.add_argument('--save_path', default="./result_png", help='path to dataset')
    parser.add_argument('--crop', default=None, type=int, nargs='+', help="crop region")
    args = parser.parse_args()
    image_processing(args)
