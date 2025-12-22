import torch
import sys
import os

print("="*40)
print("🔍 DIAGNÓSTICO DE HARDWARE (GPU/CPU)")
print("="*40)

# 1. Verificar disponibilidad de CUDA
is_cuda = torch.cuda.is_available()
print(f"⚡ CUDA Disponible: {is_cuda}")

if is_cuda:
    # 2. Detalles de la Tarjeta Gráfica
    device_count = torch.cuda.device_count()
    print(f"🔢 Cantidad de GPUs: {device_count}")
    
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   🚀 GPU {i}: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        
    # Prueba de Tensor en GPU
    try:
        x = torch.tensor([1.0, 2.0]).cuda()
        print("✅ Prueba de Tensor: ÉXITO (El tensor vive en la GPU)")
    except Exception as e:
        print(f"❌ Prueba de Tensor: FALLÓ ({e})")
else:
    print("⚠️  ADVERTENCIA: Estás corriendo en CPU. El entrenamiento será lento.")
    print("   (Asegúrate de pedir '--gres=gpu:1' en Slurm)")

print("="*40)
