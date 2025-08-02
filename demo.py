import os
import torch
import matplotlib.pyplot as plt
import numpy as np

import backbones
import simplenet
import cv2


class SimpleNetJIT(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        backbone_name = "resnet50"
        layers_to_extract_from = ["layer2", "layer3"]
        input_shape = [3, 224, 224]

        backbone = backbones.load(backbone_name)
        backbone.name = backbone_name

        self.model = simplenet.SimpleNet(device)
        self.model.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            embedding_size=256,
            meta_epochs=40,
            gan_epochs=4,
            noise_std=0.015,
            dsc_hidden=1024,
            dsc_layers=2,
            dsc_margin=0.5,
            pre_proj=1
        )


        assert os.path.exists(ckpt_path)
        print(f"Loading trained model from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        if 'discriminator' in state_dict:
            self.model.discriminator.load_state_dict(state_dict['discriminator'])
            if "pre_projection" in state_dict:
                self.model.pre_projection.load_state_dict(state_dict["pre_projection"])
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, image):
        # 模型代码的一些写法导致模型无法支持动态batch_size，这里我们显式地对输入做reshape
        # 这相当于插入了一个断言，强制了输入的batch_size
        image = image.reshape(4, 3, 224, 224)
        features, patch_shapes = self.model._embed(image, provide_patch_shapes=True,
                                                 evaluation=True)
        features = self.model.pre_projection(features)
        image_scores = -1 * self.model.discriminator(features)
        #
        patch_shape_0 = patch_shapes[0]
        image_scores = image_scores.view(image.size(0), int(patch_shape_0[0]), int(patch_shape_0[1]))
        image_scores = image_scores.unsqueeze(1)
        image_scores = torch.nn.functional.interpolate(image_scores, size=(image.shape[2], image.shape[3]), mode="bilinear")
        return image_scores
        # image_scores, masks, features = self.model._predict(image)
        # masks = torch.from_numpy(masks[0]).sigmoid()
        # return masks
        # return mage_scores, masks, features

    def eval(self):
        for module in self.model.children():
            module.train(False)

def preprocess_image(image_path):
    from torchvision import transforms
    import PIL.Image
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    image = PIL.Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    processed_image = transform(image).unsqueeze(0)  # Add batch dimension

    return processed_image

if __name__ == "__main__":
    # Example usage
    # image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/val/cropped/搭环/148-abnormal.bmp"  # Replace with actual image path
    # image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/val/cropped/搭环/174-abnormal.bmp"
    # image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/val/cropped/搭环/156-abnormal.bmp"
    # image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/val/cropped/搭环/148-abnormal.bmp"
    # image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/val/cropped/搭环/168.bmp"
    # image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/val/cropped/搭环/167.bmp"
    # image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/val/cropped/搭环/178.bmp"
    image_path = "/home/kohill/aoidev/datasets/中瓷/3D下料/train/cropped/HS20250730195927177/0.bmp"
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    ckpt_path = "/home/kohill/Desktop/workspace/SimpleNet/results/MVTecAD_Results/simplenet_mvtec/run/models/0/cofired_train/ckpt_0.pth"
    name_prefix = ckpt_path[:-4]
    # jit_output_path = name_prefix + "-jit.pth"
    onnx_output_path = name_prefix + "-onnx.onnx"
    jit_output_path = "/home/kohill/aoidev/home/kohill/tmp/0802/ckpt_4-jit.pt"
    model = SimpleNetJIT(ckpt_path=ckpt_path).to(device)
    model.eval()

    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)[:, :, ::-1]
    # image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    image_transpose = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_tensor1 = torch.from_numpy(image_transpose[np.newaxis]).to(device)
    image_tensor1 = torch.nn.functional.interpolate(image_tensor1, size=(224, 224), mode="bilinear", align_corners=False)
    image_tensor1 = torch.cat([image_tensor1, image_tensor1] * 2, dim=0)
    # image_tensor2 = preprocess_image(image_path).to(device) * 255.0
    def normalize_tensor(tensor):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(tensor.device)
        std =  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(tensor.device)
        tensor = tensor.float() / 255.0
        tensor = (tensor - mean) / std
        return tensor


    image_tensor1_normalized = normalize_tensor(image_tensor1)
    # image_tensor2_normalized = normalize_tensor(image_tensor2)
    with torch.no_grad():
        # Use the predict method directly
        masks1 = model(image_tensor1_normalized)
        # masks2 = model(image_tensor2_normalized)
        traced_model = torch.jit.trace(model, image_tensor1_normalized)
        torch.jit.save(traced_model, jit_output_path)
        # torch.onnx.export(
        #     traced_model,
        #     image_tensor1,
        #     onnx_output_path,
        #     export_params=True,
        #     opset_version=11,
        #     input_names=['input'],
        #     output_names=['output'],
        #     dynamic_axes={
        #         'input': {0: 'batch_size'},
        #         'output': {0: 'batch_size'}
        #     }
        # )
        # print(f"ONNX model exported to: {onnx_output_path}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 4))
    axes = axes.reshape(-1)
    axes[0].imshow(image_tensor1.cpu().numpy()[0].transpose((1, 2, 0)).astype(np.uint8))
    axes[1].imshow(masks1[0, 0].sigmoid().data.cpu().numpy(), vmin=0, vmax=1)
    # axes[2].imshow(image_tensor2.cpu().numpy()[0].transpose((1, 2, 0)).astype(np.uint8))
    # axes[3].imshow(masks2[0, 0].sigmoid().data.cpu().numpy(), vmin=0, vmax=1)
    plt.show()
