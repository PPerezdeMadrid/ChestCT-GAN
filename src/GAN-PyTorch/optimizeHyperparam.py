import optuna
import torch.optim as optim

def objective(trial):
    """Función de Optuna para encontrar los mejores hiperparámetros."""
    
    # Espacio de búsqueda de hiperparámetros
    lrG = trial.suggest_loguniform("lrG", 1e-5, 1e-3)
    lrD = trial.suggest_loguniform("lrD", 1e-5, 1e-3)
    beta1 = trial.suggest_float("beta1", 0.5, 0.9)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)

    # Configuración del dispositivo
    device = setup_device()
    config = load_config()
    params = config["params"]
    dataloader = get_chestct(params["imsize"])
    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
    
    # Inicializar modelo y optimizadores
    netG, netD = initialize_model('dcgan', params, device)
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, beta2))
    
    criterion = torch.nn.BCELoss()

    # Entrenar por pocas épocas para evaluar hiperparámetros
    G_losses, D_losses, _ = train_dcgan(params, dataloader, netG, netD, optimizerG, optimizerD, criterion, fixed_noise, device, model_path=".")

    return min(G_losses)  # Queremos minimizar la pérdida del generador

# Ejecutar la optimización con Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Realizar 20 pruebas

# Obtener los mejores hiperparámetros
best_params = study.best_params
print("Mejores hiperparámetros encontrados:", best_params)


"""
Nota: añadir @cards En Metaflow, la etiqueta @card se usa para generar visualizaciones y reportes en la interfaz de Metaflow UI.
 Permite adjuntar información en formato Markdown, HTML o JSON a un paso específico del flujo (Step), facilitando la inspección de r
 esultados, métricas o gráficos directamente en la UI.
"""
