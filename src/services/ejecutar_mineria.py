#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
from mineriaService import KataluMineriaService

def main():
    try:
        # Obtener DATABASE_URL desde argumentos o variable de entorno
        if len(sys.argv) > 1:
            database_url = sys.argv[1]
        else:
            database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:1234@localhost:5432/bdkatalu')
        
        # Crear instancia del servicio
        predictor = KataluMineriaService(database_url)
        
        # Ejecutar análisis completo
        result = predictor.run_full_analysis()
        
        # Imprimir resultado como JSON
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        # Imprimir error como JSON
        error_result = {
            "success": False, 
            "error": str(e)
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()