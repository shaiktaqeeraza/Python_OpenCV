import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
 
 
class ImageEditorStudio:
    MAX_PREVIEW_SIZE = (700, 700)  # (width, height) cap for on-screen preview
 
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Image Editing Studio")
        self.root.geometry("1150x780")
        self.root.minsize(950, 650)
 
        # Image state
        self.original_image = None   # BGR, untouched, full resolution
        self.working_image = None    # BGR, resized copy used for fast preview
        self.processed_image = None  # BGR, after all edits (preview resolution)
        self.tk_image = None         # PhotoImage currently shown
 
        self._build_layout()
        self._build_controls()
 
    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)
 
        ttk.Button(toolbar, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Reset", command=self.reset_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=4)
 
        self.status_var = tk.StringVar(value="No image loaded.")
        ttk.Label(toolbar, textvariable=self.status_var, foreground="gray").pack(side=tk.LEFT, padx=15)
 
        # Main body: left preview | right scrollable controls
        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
 
        # --- Left: image preview ---
        left = ttk.Frame(body, padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
 
        self.canvas_label = ttk.Label(left, text="Upload an image to begin", anchor="center",
                                       relief="solid", background="#dddddd")
        self.canvas_label.pack(fill=tk.BOTH, expand=True)
 
        # --- Right: scrollable controls panel ---
        right_container = ttk.Frame(body, width=340)
        right_container.pack(side=tk.RIGHT, fill=tk.Y)
        right_container.pack_propagate(False)
 
        control_canvas = tk.Canvas(right_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=control_canvas.yview)
        self.scrollable_frame = ttk.Frame(control_canvas)
 
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
 
        control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
 
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
        # Enable mouse-wheel scrolling over the controls panel
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
 
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)   # Windows / macOS
        control_canvas.bind_all("<Button-4>", lambda e: control_canvas.yview_scroll(-1, "units"))  # Linux
        control_canvas.bind_all("<Button-5>", lambda e: control_canvas.yview_scroll(1, "units"))
 
    # ------------------------------------------------------------------ #
    # Controls (sliders)
    # ------------------------------------------------------------------ #
    def _build_controls(self):
        frame = self.scrollable_frame
        ttk.Label(frame, text="Adjustments", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5))
 
        # Each slider entry: (label, attribute name, from, to, default, resolution)
        self.brightness_var = self._add_slider(frame, "Brightness", -100, 100, 0)
        self.contrast_var = self._add_slider(frame, "Contrast (%)", 50, 300, 100)
        self.saturation_var = self._add_slider(frame, "Saturation (%)", 0, 200, 100)
        self.hue_var = self._add_slider(frame, "Hue Shift", -180, 180, 0)
        self.blur_var = self._add_slider(frame, "Gaussian Blur", 0, 20, 0)
        self.sharpen_var = self._add_slider(frame, "Sharpen", 0, 100, 0)
        self.rotation_var = self._add_slider(frame, "Rotation (°)", 0, 360, 0)
        self.zoom_var = self._add_slider(frame, "Zoom (%)", 50, 200, 100)
 
        ttk.Separator(frame, orient="horizontal").pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frame, text="Tip: use Reset to clear all adjustments.",
                  foreground="gray", wraplength=280, justify="left").pack(anchor="w", padx=10, pady=(0, 15))
 
    def _add_slider(self, parent, label, frm, to, default):
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=10, pady=6)
 
        header = ttk.Frame(container)
        header.pack(fill=tk.X)
        ttk.Label(header, text=label).pack(side=tk.LEFT)
        value_label = ttk.Label(header, text=str(default))
        value_label.pack(side=tk.RIGHT)
 
        var = tk.DoubleVar(value=default)
 
        def on_change(_evt=None, v=var, lbl=value_label):
            lbl.config(text=str(int(v.get())))
            self.update_preview()
 
        scale = ttk.Scale(container, from_=frm, to=to, orient="horizontal",
                           variable=var, command=lambda val: on_change())
        scale.pack(fill=tk.X)
        return var
 
    # ------------------------------------------------------------------ #
    # File operations
    # ------------------------------------------------------------------ #
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
 
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Could not read the selected image file.")
            return
 
        self.original_image = image
        # Downscale a copy for smooth, responsive editing/preview
        self.working_image = self._fit_to_max(image, self.MAX_PREVIEW_SIZE)
        self.status_var.set(f"Loaded: {path.split('/')[-1]}  "
                             f"({image.shape[1]}x{image.shape[0]})")
        self.reset_sliders(update=False)
        self.update_preview()
 
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("No image", "There is no processed image to save yet.")
            return
 
        path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Bitmap", "*.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
 
        # Re-apply the full pipeline on the FULL-RESOLUTION original for a quality save
        full_res_result = self.apply_pipeline(self.original_image)
        success = cv2.imwrite(path, full_res_result)
        if success:
            messagebox.showinfo("Saved", f"Image saved to:\n{path}")
        else:
            messagebox.showerror("Error", "Failed to save the image.")
 
    def reset_image(self):
        if self.original_image is None:
            return
        self.reset_sliders(update=True)
 
    def reset_sliders(self, update=True):
        self.brightness_var.set(0)
        self.contrast_var.set(100)
        self.saturation_var.set(100)
        self.hue_var.set(0)
        self.blur_var.set(0)
        self.sharpen_var.set(0)
        self.rotation_var.set(0)
        self.zoom_var.set(100)
        if update:
            self.update_preview()
 
    # ------------------------------------------------------------------ #
    # Image processing pipeline
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_to_max(image, max_size):
        h, w = image.shape[:2]
        max_w, max_h = max_size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image
 
    def apply_pipeline(self, source_bgr):
        """Apply all slider-driven operations, in order, to the given BGR image."""
        img = source_bgr.copy()
 
        brightness = int(self.brightness_var.get())
        contrast_pct = self.contrast_var.get()
        saturation_pct = self.saturation_var.get()
        hue_shift = int(self.hue_var.get())
        blur_amount = int(self.blur_var.get())
        sharpen_amount = self.sharpen_var.get()
        rotation_deg = self.rotation_var.get()
        zoom_pct = self.zoom_var.get()
 
        # 1. Brightness & Contrast: new = alpha * img + beta
        alpha = contrast_pct / 100.0
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=brightness)
 
        # 2. Saturation & Hue via HSV
        if saturation_pct != 100 or hue_shift != 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (saturation_pct / 100.0), 0, 255)
            hsv = hsv.astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
 
        # 3. Gaussian Blur
        if blur_amount > 0:
            k = 2 * blur_amount + 1  # ensure odd kernel size
            img = cv2.GaussianBlur(img, (k, k), 0)
 
        # 4. Sharpen (unsharp mask)
        if sharpen_amount > 0:
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            amount = sharpen_amount / 100.0 * 2.0  # scale factor
            img = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
 
        # 5. Rotation (keeps full image in frame, fills background with white)
        if rotation_deg % 360 != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            cos = abs(m[0, 0])
            sin = abs(m[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            m[0, 2] += (new_w / 2) - center[0]
            m[1, 2] += (new_h / 2) - center[1]
            img = cv2.warpAffine(img, m, (new_w, new_h), borderValue=(255, 255, 255))
 
        # 6. Zoom in / out (scale, then crop or pad back to original-ish size)
        if zoom_pct != 100:
            scale = zoom_pct / 100.0
            h, w = img.shape[:2]
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
 
            if scale >= 1.0:
                # Zoom in: crop the center back to the original size
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                img = resized[start_y:start_y + h, start_x:start_x + w]
            else:
                # Zoom out: pad with white to the original size
                canvas = np.full((h, w, 3), 255, dtype=np.uint8)
                off_x = (w - new_w) // 2
                off_y = (h - new_h) // 2
                canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
                img = canvas
 
        return img
 
    def update_preview(self):
        if self.working_image is None:
            return
 
        self.processed_image = self.apply_pipeline(self.working_image)
        display_img = self._fit_to_max(self.processed_image, self.MAX_PREVIEW_SIZE)
 
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)
 
        self.canvas_label.config(image=self.tk_image, text="")
        self.canvas_label.image = self.tk_image
 
 
def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = ImageEditorStudio(root)
    root.mainloop()
 
 
if __name__ == "__main__":
    main()