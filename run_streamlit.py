import subprocess

# Caminho para o script Streamlit
script_path = "c:/Users/steve/OneDrive/Área de Trabalho/OBT/OBT/OBT/dados_pre_processados.py"

# Comando para executar o Streamlit
command = f'streamlit run "{script_path}"'

# Executa o comando
subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command], check=True)
