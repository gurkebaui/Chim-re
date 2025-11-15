import torch
from mamba_ssm import Mamba

def run_test():
    """
    Ein schnelles Skript zum Testen der PyTorch- und Mamba-SSM-Installation auf der GPU.
    """
    print("--- Test für Mamba-SSM Installation ---")

    # 1. Überprüfe die GPU-Verfügbarkeit
    if not torch.cuda.is_available():
        print("❌ FEHLER: CUDA ist nicht verfügbar. PyTorch kann die GPU nicht finden.")
        return

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU gefunden: {gpu_name}")

    try:
        # 2. Definiere Modellparameter und erstelle einen Dummy-Input
        batch_size = 4
        seq_length = 256
        d_model = 64  # Modelldimension

        # Erstelle einen zufälligen Tensor auf der GPU
        input_tensor = torch.randn(batch_size, seq_length, d_model, device=device)
        print(f"\n✅ Dummy-Input-Tensor erstellt mit Shape: {input_tensor.shape}")

        # 3. Initialisiere das Mamba-Modell
        print("   Initialisiere das Mamba-Modell...")
        model = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)
        print("✅ Mamba-Modell erfolgreich initialisiert und auf die GPU verschoben.")

        # 4. Führe einen Forward Pass durch
        print("   Führe einen Forward Pass aus...")
        # Wir brauchen keine Gradienten für diesen Test
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        print("✅ Forward Pass erfolgreich abgeschlossen.")
        print(f"✅ Output-Tensor hat den Shape: {output_tensor.shape}")

        # 5. Finale Überprüfung
        if output_tensor.shape == input_tensor.shape:
            print("\n🎉 ERFOLG! Deine PyTorch- und Mamba-SSM-Installation scheint korrekt auf der GPU zu funktionieren.")
        else:
            print(f"\n⚠️ WARNUNG: Der Output-Shape {output_tensor.shape} stimmt nicht mit dem Input-Shape {input_tensor.shape} überein.")

    except Exception as e:
        print(f"\n❌ Ein Fehler ist während des Tests aufgetreten: {e}")
        print("   Das könnte auf ein Problem mit den CUDA-Kernels, eine fehlerhafte Kompilierung oder eine Versions-Inkompatibilität hindeuten.")

if __name__ == "__main__":
    run_test()