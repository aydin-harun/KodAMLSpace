import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from bitsandbytes.nn import Linear4bit
import os, gc, cv2, base64, io
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from libs.utils.configReader import EmptyPageModelConfig
import gc


# === Quantized CNN (GPU + 4-bit) ===
class QuantizedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = Linear4bit(64 * 64 * 64, 128, bias=True)
        self.fc2 = Linear4bit(128, 2, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# === Simple CNN (CPU) ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# === Lazy Loading Dataset ===
class LazyImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path, label = self.file_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.fastNlMeansDenoising(img, h=30)
        dpi_scale = 100 / 300  # 300 dpi -> 100 dpi ölçekleme
        resized = cv2.resize(img, None, fx=dpi_scale, fy=dpi_scale, interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        img_tensor = self.transform(pil_img) if self.transform else pil_img
        del img, resized, pil_img
        return img_tensor, label


# === Detector Class ===
class CnnEmptyPageDetector:
    def __init__(self, emptyPageModelConfig: EmptyPageModelConfig):
        self.emptyPageModelConfig = emptyPageModelConfig
        self.model_path = emptyPageModelConfig.cnnModelPath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QuantizedCNN().to(self.device) if torch.cuda.is_available() else SimpleCNN().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.accelerator = Accelerator()
        self.isModelLoad = False
        print(f"🔥 Device: {self.device} | Model Type: {'QuantizedCNN (4bit)' if torch.cuda.is_available() else 'SimpleCNN'}")

    def _decode_base64_image(self, b64_str):
        img_data = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        img = cv2.fastNlMeansDenoising(img, h=30)
        dpi_scale = 100 / 300
        resized = cv2.resize(img, None, fx=dpi_scale, fy=dpi_scale, interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        del img_data, np_arr, img, resized, pil_img
        return input_tensor

    def train(self, empty_dir, filled_dir, batch_size=8, epochs=10, lr=0.001):
        file_paths = [(os.path.join(folder, f), label)
                      for label, folder in enumerate([empty_dir, filled_dir])
                      for f in os.listdir(folder)
                      if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

        X_train, X_test = train_test_split(file_paths, test_size=0.2, stratify=[lbl for _, lbl in file_paths])

        test_dataset = LazyImageDataset(X_test, self.transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model, test_loader = self.accelerator.prepare(self.model, test_loader)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            train_dataset = LazyImageDataset(X_train, self.transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            train_loader = self.accelerator.prepare(train_loader)

            model.train()
            running_loss = 0.0
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")

            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                self.accelerator.backward(loss)
                optimizer.step()
                running_loss += loss.item()
                percent = ((batch_idx + 1) / len(train_loader)) * 100
                print(f"    > %{percent:.1f} ({batch_idx + 1}/{len(train_loader)})", end='\r')

            print(f"\n    ✔️ Epoch tamamlandı | Loss: {running_loss / len(train_loader):.4f}")
            del train_loader, train_dataset
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        torch.save(self.accelerator.unwrap_model(self.model).state_dict(), self.model_path)
        print(f"✅ Model kaydedildi: {self.model_path}")

        del model, test_loader, X_train, X_test
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.isModelLoad = False
        return self.model_path

    def predict_from_base64(self, b64_str):
        if not self.isModelLoad:
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.isModelLoad = True
            print("📦 Model başarıyla yüklendi ve hazır.")

        input_tensor = self._decode_base64_image(b64_str)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            prob = torch.softmax(output, dim=1)[0][predicted].item() * 100

        result = "Dolu" if predicted.item() == 1 else "Boş"
        print(f"➜ Tahmin: {result} (%{prob:.2f} güven)")
        remark, clas = self.getDescriptions(prob, predicted.item())

        del input_tensor, output
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return {
            "pred": int(predicted.item()),
            "prob": float(prob),
            "clas": clas,
            "remark": remark,
            "lib_type": "cnn_4bit" if torch.cuda.is_available() else "cnn_cpu"
        }

    def getDescriptions(self, prob, pred):
        remark = "Kararsız" if prob < 60 else "Orta Güven" if prob < 85 else "Yüksek Güven"
        clas = "Dolu" if pred == 1 else "Boş"
        return remark, clas

    def print_model_summary(self):
        print("📊 MODEL ÖZETİ")
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Toplam Parametre: {total_params:,}")
        print(f"Eğitilebilir Parametre: {trainable_params:,}")

    def checkModelExists(self):
        return  len(self.model_path) > 0 and os.path.exists(self.model_path)


##########################################11
# # === Quantized CNN (GPU + 4-bit) ===
# class QuantizedCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = Linear4bit(64 * 128 * 128, 128, bias=True)
#         self.fc2 = Linear4bit(128, 2, bias=True)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 128 * 128)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # === Simple CNN (CPU) ===
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 128 * 128, 128)
#         self.fc2 = nn.Linear(128, 2)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 128 * 128)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # === Dataset ===
# class ImageDataset(Dataset):
#     def __init__(self, file_paths, transform=None):
#         self.file_paths = file_paths  # (path, label)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         path, label = self.file_paths[idx]
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.fastNlMeansDenoising(img, h=30)
#         pil_img = Image.fromarray(img)
#         del img
#         if self.transform:
#             img_tensor = self.transform(pil_img)
#         else:
#             img_tensor = pil_img
#         del pil_img
#         return img_tensor, label
#
#
# # === Detector Class ===
# class CnnEmptyPageDetector:
#     def __init__(self, emptyPageModelConfig: EmptyPageModelConfig):
#         self.emptyPageModelConfig = emptyPageModelConfig
#         self.model_path = emptyPageModelConfig.cnnModelPath
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = QuantizedCNN().to(self.device) if torch.cuda.is_available() else SimpleCNN().to(self.device)
#         self.transform = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#         self.accelerator = Accelerator()
#         self.isModelLoad = False
#
#     def _decode_base64_image(self, b64_str):
#         img_data = base64.b64decode(b64_str)
#         np_arr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
#         img = cv2.fastNlMeansDenoising(img, h=30)
#         pil_img = Image.fromarray(img)
#         input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
#         # Bellek temizliği
#         del img_data, np_arr, img, pil_img
#         return input_tensor
#
#     def train(self, empty_dir, filled_dir, batch_size=8, epochs=10, lr=0.001):
#         file_paths = []
#         for label, folder in enumerate([empty_dir, filled_dir]):
#             for file in os.listdir(folder):
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
#                     file_paths.append((os.path.join(folder, file), label))
#
#         total_files = len(file_paths)
#         print(f"📁 Toplam eğitim verisi: {total_files} dosya")
#
#         # === % ilerleyiş raporu ===
#         # progress_step = max(1, total_files // 100)
#         # next_progress = progress_step
#         # for idx, _ in enumerate(file_paths):
#         #     if (idx + 1) >= next_progress or (idx + 1) == total_files:
#         #         percent = ((idx + 1) / total_files) * 100
#         #         print(f"    > %{percent:.1f} işlenecek ({idx + 1}/{total_files})")
#         #         next_progress += progress_step
#         #         gc.collect()
#
#         X_train, X_test = train_test_split(file_paths, test_size=0.2, stratify=[lbl for _, lbl in file_paths])
#
#         train_dataset = ImageDataset(X_train, transform=self.transform)
#         test_dataset = ImageDataset(X_test, transform=self.transform)
#
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#         model, train_loader, test_loader = self.accelerator.prepare(self.model, train_loader, test_loader)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#         model.train()
#         for epoch in range(epochs):
#             running_loss = 0.0
#             total_batches = len(train_loader)
#             print(f"📚 Epoch {epoch + 1}/{epochs}")
#
#             for batch_idx, (images, labels) in enumerate(train_loader):
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 self.accelerator.backward(loss)
#                 optimizer.step()
#                 running_loss += loss.item()
#
#                 # 👇 İlerleme oranı
#                 percent = ((batch_idx + 1) / total_batches) * 100
#                 print(f"    > Epoch ilerlemesi: %{percent:.1f} ({batch_idx + 1}/{total_batches})", end='\r')
#
#             print(f"\n    ✔️ Epoch {epoch + 1} tamamlandı | Loss: {running_loss / total_batches:.4f}")
#             gc.collect()
#
#         torch.save(self.model.state_dict(), self.model_path)
#         print(f"✅ Model kaydedildi: {self.model_path}")
#
#         del model, train_loader, test_loader
#         del file_paths, X_train, X_test, train_dataset, test_dataset
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         self.isModelLoad = False
#         return self.model_path
#
#     def predict_from_base64(self, b64_str):
#         if not self.isModelLoad and (len(self.model_path) > 0 and os.path.exists(self.model_path)):
#             self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
#             self.model.to(self.device)
#             self.model.eval()
#             self.isModelLoad = True
#             self.print_model_info()
#         input_tensor = self._decode_base64_image(b64_str)
#         with torch.no_grad():
#             output = self.model(input_tensor)
#             _, predicted = torch.max(output, 1)
#             prob = torch.softmax(output, dim=1)[0][predicted].item() * 100
#
#         result = "Dolu" if predicted.item() == 1 else "Boş"
#         print(f"➜ Tahmin: {result} (%{prob:.2f} güven)")
#         remark, clas = self.getDescriptions(prob, predicted.item())
#
#         print(f"🔥 input_tensor device: {input_tensor.device}")
#         print(f"🔥 model first param device: {next(self.model.parameters()).device}")
#         print(f"🔥 fc1 type: {type(self.model.fc1)}")
#         print(f"🔥 fc2 type: {type(self.model.fc2)}")
#         if torch.cuda.is_available():
#             print(f"🔥 CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
#             print(f"🔥 CUDA memory reserved : {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
#
#         del input_tensor, output
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         return {
#             "pred": int(predicted.item()),
#             "prob": float(prob),
#             "clas": clas,
#             "remark": remark,
#             "lib_type": "cnn_4bit" if torch.cuda.is_available() else "cnn_cpu"
#         }
#
#     def getDescriptions(self, prob, pred):
#         if prob < 60:
#             remark = "Kararsız"
#         elif prob < 85:
#             remark = "Orta Güven"
#         else:
#             remark = "Yüksek Güven"
#         clas = "Dolu" if pred == 1 else "Boş"
#         return remark, clas
#
#     def checkModelExists(self):
#         return  len(self.model_path) > 0 and os.path.exists(self.model_path)
#
#     def print_model_info(self):
#         model_type = self.model.__class__.__name__
#         device_type = "GPU" if torch.cuda.is_available() else "CPU"
#
#         total_params = sum(p.numel() for p in self.model.parameters())
#         trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         total_layers = sum(1 for _ in self.model.modules())
#
#         param_size_bytes = sum(p.element_size() * p.numel() for p in self.model.parameters())
#         param_size_mb = param_size_bytes / (1024 ** 2)
#
#         print("🧠 Model Bilgisi")
#         print(f"   • Model tipi           : {model_type}")
#         print(f"   • Cihaz                : {device_type}")
#         print(f"   • Toplam parametre     : {total_params:,}")
#         print(f"   • Eğitilebilir parametre : {trainable_params:,}")
#         print(f"   • Katman sayısı        : {total_layers}")
#         print(f"   • Yaklaşık model boyutu: {param_size_mb:.2f} MB")








###################################################2222222
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from accelerate import Accelerator
# from bitsandbytes.nn import Linear4bit
# import os, gc, cv2, base64
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from libs.utils.configReader import EmptyPageModelConfig
#
#
# # === Quantized CNN (GPU + 4-bit) ===
# class QuantizedCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = Linear4bit(64 * 128 * 128, 128, bias=True)
#         self.fc2 = Linear4bit(128, 2, bias=True)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 128 * 128)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # === Simple CNN (CPU) ===
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 128 * 128, 128)
#         self.fc2 = nn.Linear(128, 2)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 128 * 128)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # === Dataset ===
# class ImageDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#         self.images = images
#         self.labels = labels
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, self.labels[idx]
#
#
# # === Detector Class ===
# class CnnEmptyPageDetector:
#     def __init__(self, emptyPageModelConfig: EmptyPageModelConfig):
#         self.emptyPageModelConfig = emptyPageModelConfig
#         self.model_path = emptyPageModelConfig.cnnModelPath
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = QuantizedCNN().to(self.device) if torch.cuda.is_available() else SimpleCNN().to(self.device)
#         self.transform = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#         self.accelerator = Accelerator()
#         self.isModelLoad = False
#
#     def _decode_base64_image(self, b64_str):
#         img_data = base64.b64decode(b64_str)
#         np_arr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
#         img = cv2.fastNlMeansDenoising(img, h=30)
#         pil_img = Image.fromarray(img)
#         return self.transform(pil_img).unsqueeze(0).to(self.device)
#
#     def train(self, empty_dir, filled_dir, batch_size=8, epochs=10, lr=0.001):
#         images, labels, file_paths = [], [], []
#
#         for label, folder in enumerate([empty_dir, filled_dir]):
#             for file in os.listdir(folder):
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
#                     file_paths.append((os.path.join(folder, file), label))
#
#         total_files = len(file_paths)
#         progress_step = max(1, total_files // 100)
#         next_progress = progress_step
#         print(f"📁 Toplam eğitim verisi: {total_files} dosya")
#
#         for idx, (path, label) in enumerate(file_paths):
#             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             img = cv2.fastNlMeansDenoising(img, h=30)
#             pil_img = Image.fromarray(img)
#             images.append(pil_img)
#             labels.append(label)
#
#             if (idx + 1) >= next_progress or (idx + 1) == total_files:
#                 percent = ((idx + 1) / total_files) * 100
#                 print(f"    > %{percent:.1f} tamamlandı ({idx + 1}/{total_files})")
#                 next_progress += progress_step
#
#         X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)
#         train_dataset = ImageDataset(X_train, y_train, transform=self.transform)
#         test_dataset = ImageDataset(X_test, y_test, transform=self.transform)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#         model, train_loader, test_loader = self.accelerator.prepare(self.model, train_loader, test_loader)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#         model.train()
#         for epoch in range(epochs):
#             running_loss = 0.0
#             for images, labels in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 self.accelerator.backward(loss)
#                 optimizer.step()
#                 running_loss += loss.item()
#             print(f"[{epoch + 1}/{epochs}] Loss: {running_loss / len(train_loader):.4f}")
#
#         torch.save(self.model.state_dict(), self.model_path)
#         print(f"✅ Model kaydedildi: {self.model_path}")
#
#         del model, train_loader, test_loader
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         return self.model_path
#
#     def predict_from_base64(self, b64_str):
#         if not self.isModelLoad:
#             self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
#             self.model.eval()
#             self.isModelLoad = True
#
#         input_tensor = self._decode_base64_image(b64_str)
#         with torch.no_grad():
#             output = self.model(input_tensor)
#             _, predicted = torch.max(output, 1)
#             prob = torch.softmax(output, dim=1)[0][predicted].item() * 100
#
#         result = "Dolu" if predicted.item() == 1 else "Boş"
#         print(f"➜ Tahmin: {result} (%{prob:.2f} güven)")
#         remark, clas = self.getDescriptions(prob, predicted.item())
#
#         del input_tensor, output
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         return {
#             "pred": int(predicted.item()),
#             "prob": float(prob),
#             "clas": clas,
#             "remark": remark,
#             "lib_type": "cnn_4bit" if torch.cuda.is_available() else "cnn_cpu"
#         }
#
#     def getDescriptions(self, prob, pred):
#         if prob < 60:
#             remark = "Kararsız"
#         elif prob < 85:
#             remark = "Orta Güven"
#         else:
#             remark = "Yüksek Güven"
#         clas = "Dolu" if pred == 1 else "Boş"
#         return remark, clas
