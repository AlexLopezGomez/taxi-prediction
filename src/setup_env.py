# setup_env.py
import sys
import os

def setup_project_environment():
    """Configura el entorno para usar el proyecto correcto"""
    current_project_root = "/mnt/c/Users/alexl/OneDrive/Escritorio/my-version-taxi_demand_predictor"
    
    # Limpiar rutas del proyecto original
    sys.path = [path for path in sys.path 
               if not ('taxi_demand_predictor' in path and 'my-version-taxi_demand_predictor' not in path)]
    
    # Limpiar módulos cacheados
    modules_to_remove = [mod for mod in list(sys.modules.keys()) if mod.startswith('src.')]
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]
    
    # Agregar la ruta correcta
    if current_project_root not in sys.path:
        sys.path.insert(0, current_project_root)
    
    print(f"✅ Entorno configurado para: {current_project_root}")
    return current_project_root

if __name__ == "__main__":
    setup_project_environment()