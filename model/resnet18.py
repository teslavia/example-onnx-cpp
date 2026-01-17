import torch
import torchvision.models as models

def export_onnx():
    # 1. 加载预训练的 ResNet-18 (安防常用轻量级 backbone)
    # weights='DEFAULT' 会加载 ImageNet 预训练权重
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval() # 必须切换到 eval 模式，固化 BatchNorm 和 Dropout

    # 2. 定义虚拟输入 (Batch_Size, Channel, Height, Width)
    # 安防中常用 224x224 或 256x128(ReID)，这里用标准的 ImageNet 尺寸
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. 指定输入输出节点名称（方便 C++ 调用）
    input_names = ["input_image"]
    output_names = ["class_logits"]

    # 4. 导出 ONNX
    output_path = "../model/out/resnet18.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=13, # 建议使用 11 或 13，兼容性较好
        # 动态轴：允许 C++ 推理时改变 Batch Size
        dynamic_axes={"input_image": {0: "batch_size"}, "class_logits": {0: "batch_size"}}
    )
    
    print(f"Model exported to {output_path}")

    # 5. (可选) 验证导出是否成功
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX validity check passed!")
    except ImportError:
        print("Install 'onnx' pip package to verify.")

if __name__ == "__main__":
    export_onnx()