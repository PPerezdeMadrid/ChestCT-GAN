import optuna
import json
import GAN_PyTorch.train_pipeline as train
from fpdf import FPDF

def objective(trial, params, model_type, dataset, study_logs):
    # Valores para optimizar
    params["bsize"] = trial.suggest_int("bsize", 32, 256) 
    params["nz"] = trial.suggest_int("nz", 50, 200)  # Tamaño del vector latente
    params["ngf"] = trial.suggest_int("ngf", 64, 256)  # Número de filtros en la red generadora
    params["ndf"] = trial.suggest_int("ndf", 64, 256)  # Número de filtros en la red discriminadora
    # params["nepochs"] = trial.suggest_int("nepochs", 100, 2000)  
    params["nepochs"] = 2
    params["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-2)  
    params["beta1"] = trial.suggest_uniform("beta1", 0.0, 0.9)  
    params["beta2"] = trial.suggest_uniform("beta2", 0.9, 0.999)  
    params["imsize"] = 64
    params["nc"] = 1
    params["save_epoch"] = params["nepochs"]/2
    
    arg = {
            'model_type': model_type,
            'dataset': dataset,
        }
    
    _, _, log_csv_path = train.main(arg, params)
    
    # Evaluar rendimiento (Tomar última pérdida de G)
    with open(log_csv_path, "r") as f:
        last_line = f.readlines()[-1]
        loss_g = float(last_line.split(",")[5]) 
    
    # Guardar los logs para cada intento
    study_logs.append({
        'trial': trial.number,
        'params': params,
        'loss_g': loss_g
    })
    
    return loss_g  # Minimizar la pérdida del generador

def generate_pdf(study_logs, best_params, pdf_name, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Título
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Optimización de Parámetros - Resultados", ln=True, align="C")
    
    # Logs de intentos
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Logs de los intentos de optimización:")
    
    # Detalles de cada intento
    pdf.set_font("Courier", size=10)
    for log in study_logs:
        pdf.multi_cell(0, 10, txt=f"Intento #{log['trial']} - Pérdida: {log['loss_g']:.4f}")
        for param, value in log['params'].items():
            pdf.multi_cell(0, 10, txt=f"  {param}: {value}")
        pdf.ln(5)
    
    # Mejores parámetros
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Mejores Parámetros Encontrados:", ln=True)
    pdf.set_font("Courier", size=12)
    pdf.multi_cell(0, 10, txt=json.dumps(best_params, indent=4))
    
    # Guardar el archivo PDF
    full_path = f"{pdf_path}/{pdf_name}.pdf"
    pdf.output(full_path)
    print(f"PDF generado correctamente: {full_path}")

def main(pdf_name, pdf_path, model_type, dataset):
    study_logs = []  
    study = optuna.create_study(direction="minimize")  # Minimizar loss_g
    study.optimize(lambda trial: objective(trial, {}, model_type, dataset, study_logs), n_trials=15)  # Ejecutar 15 pruebas
    
    # Guardar mejores parámetros
    best_params = study.best_params
    with open(f"{pdf_path}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    print("Mejores parámetros encontrados:")
    print(best_params)
    
    # Generar el PDF con los logs y los mejores parámetros
    generate_pdf(study_logs, best_params, pdf_name, pdf_path)
