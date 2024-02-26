import tkinter as tk
import subprocess
from tkinter import messagebox
import platform
import os


def run_script(script_name):
    script_path = f"./code/{script_name}"  # Chemin relatif vers le script dans le dossier code
    python_executable = "python3" if platform.system() != "Windows" else "python"

    try:
        print(f"Tentative d'exécution du script à partir de : {script_path}")  # Imprime le chemin pour le débogage
        print(os.listdir())

        # Exécution du script dans un processus séparé
        print("subprocess : ",python_executable,script_path)
        subprocess.run([python_executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite lors de l'exécution du script {script_name}: {e}")
    except FileNotFoundError:
        messagebox.showerror("Erreur", f"Le script {script_name} n'a pas été trouvé.")


# Configuration du style des widgets en utilisant tk
def setup_style(root):
    root.configure(bg='#f7f7f7')  # Couleur de fond de la fenêtre principale

    # Couleurs des boutons
    button_colors = {
        "Transformer Video": "#007bff",
        "CNN Letter": "#28a745",
        "Transformer Letter": "#dc3545"
    }
    return button_colors

# Initialisation de la fenêtre principale
root = tk.Tk()
root.title("Démonstrateur PER")
root.geometry("800x300")  # Ajustement de la taille pour mieux s'adapter aux éléments

button_colors = setup_style(root)

# Ajout des boutons et des étiquettes explicatives
def add_button_with_label(text, explanation, command, row, color):
    btn = tk.Button(root, text=text, command=lambda: run_script(command), bg=color, fg='white', font=('Helvetica', 12, 'bold'))
    btn.grid(row=row, column=0, pady=10, padx=20, sticky='EW')

    lbl = tk.Label(root, text=explanation, anchor='w', bg='#f7f7f7', fg='#343a40')
    lbl.grid(row=row, column=1, padx=10, sticky='W')



add_button_with_label("CNN Letter", "Reconnaissance des lettres avec notre modèle de CNN. Appuyez sur q pour arrêter", 'run_cnn.py', 1, button_colors["CNN Letter"])
add_button_with_label("Transformer Letter", "Reconnaissance des lettres avec notre modèle de Transformer. Appuyez sur q pour arrêter", 'run_transform_letter.py', 2, button_colors["Transformer Letter"])
add_button_with_label("Transformer Word", "Reconnaissance de mots avec notre modèle de Transformer et MediaPipe", 'run_word.py', 3, button_colors["Transformer Video"])



launch_warning_text = "Certains scripts peuvent mettre quelques secondes à se lancer."
launch_warning_label = tk.Label(root, text=launch_warning_text, bg='#f7f7f7', fg='#343a40', font=('Helvetica', 9))
launch_warning_label.grid(row=4, column=0, columnspan=2, pady=(30, 0), sticky='S')

# Ajout des crédits avec style et plus bas
credits_text = "RAZANAKOTO Tsiory - M'NASRI Yasmina - ROCHE Alexis"
credits_label = tk.Label(root, text=credits_text, bg='#f7f7f7', fg='#6c757d', font=('Helvetica', 9, 'italic'))
credits_label.grid(row=5, column=0, columnspan=2, pady=(0, 10), sticky='S')



root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)

# Lancement de la boucle principale
root.mainloop()
