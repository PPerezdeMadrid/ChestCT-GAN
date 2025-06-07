import optuna
import json
import GAN_PyTorch.train_pipeline as train
import GAN_PyTorch.eval_model_pipeline as eval_model
from fpdf import FPDF

def objective(trial, base_params, model_type, dataset, study_logs, final_model_name, img_ref_path):
     # Clonar los parámetros base para este trial
    params = base_params.copy()

    # Valores para optimizar
    params["bsize"] = trial.suggest_int("bsize", 32, 256) 
    # params["nz"] = trial.suggest_int("nz", 50, 200)  
    params["ngf"] = trial.suggest_int("ngf", 64, 256)  # Número de filtros en la red generadora
    params["ndf"] = trial.suggest_int("ndf", 64, 256)  # Número de filtros en la red discriminadora
    # params["nepochs"] = trial.suggest_int("nepochs", 100, 2000)  
    params["nepochs"] = 1000
    params["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-2)  
    params["beta1"] = trial.suggest_uniform("beta1", 0.0, 0.9)  
    params["beta2"] = trial.suggest_uniform("beta2", 0.9, 0.999)  
    params["save_epoch"] = params["nepochs"]/2
    
    arg = {
            'model_type': model_type,
            'dataset': dataset,

        }
    

    
    _, _, log_csv_name = train.main(arg, params)
    
    with open('GAN_PyTorch/config.json', 'r') as json_file:
        config = json.load(json_file)
    
    log_csv_path = config["model"][f'evaluation_{model_type}']

    # Coger última pérdida de G
    with open(f'{log_csv_path}/{log_csv_name}', "r") as f:
        last_line = f.readlines()[-1]
        loss_g = float(last_line.split(",")[5]) 

    
    # Evaluar el modelo
    accuracy_discriminator, accuracy_generator, ssim_score, psnr_score, lpips_score, eval_md_path = eval_model.main(model_type=model_type,dataset=dataset, final_model_name=final_model_name, img_ref_path=img_ref_path)

     # Guardar los logs para cada intento
    study_logs.append({
        'trial': trial.number,
        'params': params,
        'loss_g': loss_g,
        'lpips_score': lpips_score,
        
    })
    return lpips_score  # Minimizar el LPIPS 

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
        pdf.multi_cell(0, 10, txt=f"Intento #{log['trial']} - LPIPS: {log['lpips_score']:.4f}")
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

def main(pdf_name, pdf_path, model_type, dataset, final_model_name, img_ref_path, n_trials=15):
    study_logs = []  
    # Parámetros base que no se optimizan
    base_params = {
        "nepochs": 2,
        "nz": 100, 
        "nc": 1, 
        "imsize": 64
    }
    study = optuna.create_study(direction="minimize")  # Minimizar loss_g
    study.optimize(lambda trial: objective(trial, base_params, model_type, dataset, study_logs, final_model_name, img_ref_path), n_trials=n_trials)
    
    # Guardar mejores parámetros

    best_params = study.best_params
    with open(f"{pdf_path}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        print("Mejores parámetros encontrados:")
        print(best_params)
        print(f"Mejores parámetros guardados en: {pdf_path}/best_params.json")

    
    # Generar el PDF con los logs y los mejores parámetros
    generate_pdf(study_logs, best_params, pdf_name, pdf_path)
