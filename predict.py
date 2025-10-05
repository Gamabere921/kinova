# predict.py
import argparse
import torch
import cv2
import os
from torchvision import transforms
from PIL import Image
from models.classifier import GarbageClassifier
import yaml

# ======== Cargar config ========
with open("configs/data.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Transformación ========
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ======== Cargar modelo entrenado ========
def load_model(weights_path):
    model = GarbageClassifier(config["model_name"], num_classes=config["num_classes"])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ======== Predicción ========
def predict_image(model, img_path, class_names):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    return class_names[pred_idx], probs


# ======== Modo cámara ========
def predict_camera(model, class_names):
    cap = cv2.VideoCapture(0)  # 0 = webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = probs.argmax()

        label = f"{class_names[pred_idx]} ({probs[pred_idx]*100:.1f}%)"
        cv2.putText(frame, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.imshow("Clasificador de basura ♻️", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======== Main ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/efficientnet_b0_best.pth", help="Ruta al modelo entrenado")
    parser.add_argument("--image", type=str, help="Ruta a una imagen")
    parser.add_argument("--dir", type=str, help="Ruta a una carpeta con imágenes")
    parser.add_argument("--camera", action="store_true", help="Usar cámara en vivo")
    args = parser.parse_args()

    # Tus clases (ajústalas a las de tu dataset real)
    class_names = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

    model = load_model(args.weights)

    if args.image:
        label, probs = predict_image(model, args.image, class_names)
        print(f"Predicción: {label}")
        for i, cls in enumerate(class_names):
            print(f"  {cls}: {probs[i]*100:.2f}%")

    elif args.dir:
        for fname in os.listdir(args.dir):
            if fname.lower().endswith((".jpg",".png",".jpeg")):
                path = os.path.join(args.dir, fname)
                label, probs = predict_image(model, path, class_names)
                print(f"{fname} -> {label}")

    elif args.camera:
        predict_camera(model, class_names)

    else:
        print("❌ Usa --image, --dir o --camera")
