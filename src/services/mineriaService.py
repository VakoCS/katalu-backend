import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine, text
import warnings
import base64
import io
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class KataluMineriaService:
    def __init__(self, database_url):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.data = None
        self.model = LinearRegression()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def extract_bar_sales_data(self):
        """Extrae datos de ventas de chocolates en barra"""
        query = """
        SELECT 
            s."fecha",
            EXTRACT(MONTH FROM s."fecha") as mes,
            EXTRACT(YEAR FROM s."fecha") as año,
            EXTRACT(QUARTER FROM s."fecha") as trimestre,
            ps."cantidad" as cantidad_vendida,
            c."nombre" as categoria,
            p."nombre" as producto,
            r."nombre" as region,
            pr."nombre" as provincia,
            d."nombre" as distrito,
            CASE 
                WHEN EXTRACT(MONTH FROM s."fecha") IN (12, 1, 2) THEN 'Verano'
                WHEN EXTRACT(MONTH FROM s."fecha") IN (3, 4, 5) THEN 'Otoño'
                WHEN EXTRACT(MONTH FROM s."fecha") IN (6, 7, 8) THEN 'Invierno'
                ELSE 'Primavera'
            END as estacion
        FROM "Salida" s
        INNER JOIN "ProductoSalida" ps ON s."idSalida" = ps."idSalida"
        INNER JOIN "Producto" p ON ps."idProducto" = p."idProducto"
        INNER JOIN "Categoria" c ON p."idCategoria" = c."idCategoria"
        LEFT JOIN "Cliente" cl ON s."idCliente" = cl."idCliente"
        LEFT JOIN "Ubicacion" u ON cl."idUbicacion" = u."idUbicacion"
        LEFT JOIN "Distrito" d ON u."idDistrito" = d."idDistrito"
        LEFT JOIN "Provincia" pr ON d."idProvincia" = pr."idProvincia"
        LEFT JOIN "Region" r ON pr."idRegion" = r."idRegion"
        WHERE (UPPER(c.nombre) LIKE '%BARRA%' OR UPPER(p.nombre) LIKE '%BARRA%')
          AND r."nombre" IN ('Huánuco', 'Lima', 'Lambayeque')
          AND ps."cantidad" > 0
        ORDER BY s."fecha"
        """
        
        try:
            self.data = pd.read_sql_query(text(query), self.engine)
            return len(self.data) > 0
        except Exception as e:
            print(f"Error al extraer datos: {e}")
            return False
    
    def create_quantity_features(self):
        """Crea características para predicción"""
        self.data['fecha'] = pd.to_datetime(self.data['fecha'])
        
        self.data_monthly = self.data.groupby([
            'año', 'mes', 'trimestre', 'estacion', 'region'
        ]).agg({
            'cantidad_vendida': ['sum', 'count', 'mean']
        }).reset_index()
        
        self.data_monthly.columns = [
            '_'.join(col).strip() if col[1] else col[0] 
            for col in self.data_monthly.columns.values
        ]
        
        self.data_monthly = self.data_monthly.rename(columns={
            'cantidad_vendida_sum': 'cantidad_total_mes',
            'cantidad_vendida_count': 'num_ventas_mes', 
            'cantidad_vendida_mean': 'cantidad_promedio_venta'
        })
        
        self.data_monthly['cantidad_por_venta'] = (
            self.data_monthly['cantidad_total_mes'] / self.data_monthly['num_ventas_mes']
        )
        
        self.data_monthly['es_inicio_año'] = (self.data_monthly['mes'] <= 3).astype(int)
        self.data_monthly['es_fin_año'] = (self.data_monthly['mes'] >= 10).astype(int)
        self.data_monthly['es_medio_año'] = (
            (self.data_monthly['mes'] >= 4) & (self.data_monthly['mes'] <= 9)
        ).astype(int)
        
        self.data_monthly = self.data_monthly.sort_values(['region', 'año', 'mes'])
        
        self.data_monthly['cantidad_mes_anterior'] = (
            self.data_monthly.groupby('region')['cantidad_total_mes'].shift(1)
        )
        
        self.data_monthly['promedio_movil_cantidad_3m'] = (
            self.data_monthly.groupby('region')['cantidad_total_mes']
            .rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        )
        
        self.data_monthly['cantidad_mes_anterior'] = (
            self.data_monthly['cantidad_mes_anterior'].fillna(
                self.data_monthly['cantidad_total_mes']
            )
        )
        
        return True
    
    def prepare_linear_features(self):
        """Prepara características para Linear Regression"""
        numeric_features = [
            'mes', 'trimestre', 'num_ventas_mes',
            'cantidad_promedio_venta', 'cantidad_por_venta',
            'es_inicio_año', 'es_fin_año', 'es_medio_año',
            'cantidad_mes_anterior', 'promedio_movil_cantidad_3m'
        ]
        
        categorical_features = ['region', 'estacion']
        
        X = self.data_monthly[numeric_features + categorical_features].copy()
        y = self.data_monthly['cantidad_total_mes'].copy()
        
        for column in categorical_features:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            self.label_encoders[column] = le
        
        X = X.fillna(X.mean())
        
        return X, y
    
    def train_linear_model(self):
        """Entrena el modelo Linear Regression"""
        X, y = self.prepare_linear_features()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        mae = mean_absolute_error(self.y_test, y_test_pred)
        
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, 
            cv=5, scoring='r2'
        )
        
        return train_r2, test_r2, rmse, mae, cv_scores
    
    def predict_quantity_by_region(self, mes, año=2025):
        """Predice cantidad de ventas por región"""
        regiones = ['Huánuco', 'Lima', 'Lambayeque']
        predicciones = {}
        
        for region in regiones:
            try:
                region_encoded = self.label_encoders['region'].transform([region])[0]
                
                if mes in [12, 1, 2]:
                    estacion = 'Verano'
                elif mes in [3, 4, 5]:
                    estacion = 'Otoño'
                elif mes in [6, 7, 8]:
                    estacion = 'Invierno'
                else:
                    estacion = 'Primavera'
                
                estacion_encoded = self.label_encoders['estacion'].transform([estacion])[0]
                trimestre = (mes - 1) // 3 + 1
                
                if isinstance(self.data_monthly['region'].iloc[0], str):
                    region_data = self.data_monthly[self.data_monthly['region'] == region]
                else:
                    region_data = self.data_monthly[self.data_monthly['region'] == region_encoded]
                
                if len(region_data) > 0:
                    num_ventas_mes = region_data['num_ventas_mes'].mean()
                    cantidad_promedio_venta = region_data['cantidad_promedio_venta'].mean()
                    cantidad_por_venta = region_data['cantidad_por_venta'].mean()
                    cantidad_mes_anterior = region_data['cantidad_total_mes'].iloc[-1]
                    promedio_movil_cantidad_3m = region_data['cantidad_total_mes'].tail(3).mean()
                else:
                    num_ventas_mes = self.data_monthly['num_ventas_mes'].mean()
                    cantidad_promedio_venta = self.data_monthly['cantidad_promedio_venta'].mean()
                    cantidad_por_venta = self.data_monthly['cantidad_por_venta'].mean()
                    cantidad_mes_anterior = self.data_monthly['cantidad_total_mes'].mean()
                    promedio_movil_cantidad_3m = self.data_monthly['cantidad_total_mes'].mean()
                
                features = [
                    mes, trimestre, num_ventas_mes, cantidad_promedio_venta,
                    cantidad_por_venta, 1 if mes <= 3 else 0, 1 if mes >= 10 else 0,
                    1 if 4 <= mes <= 9 else 0, cantidad_mes_anterior,
                    promedio_movil_cantidad_3m, region_encoded, estacion_encoded
                ]
                
                X_pred = np.array([features])
                X_pred_scaled = self.scaler.transform(X_pred)
                
                prediccion = self.model.predict(X_pred_scaled)[0]
                predicciones[region] = max(0, int(prediccion))
                
            except Exception as e:
                predicciones[region] = 0
        
        return predicciones
    
    def generate_graphs(self):
        """Genera los 5 gráficos y los convierte a base64"""
        graphs = {}
        y_pred = self.model.predict(self.X_test_scaled)
        residuals = self.y_test - y_pred
        
        # 1. Real vs Predicho
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.7, s=100, color='skyblue')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Cantidad Real (unidades)')
        plt.ylabel('Cantidad Predicha (unidades)')
        plt.title('Cantidad Real vs Predicha')
        plt.grid(True, alpha=0.3)
        graphs['real_vs_predicho'] = self._plot_to_base64()
        
        # 2. Residuos
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.7, s=100, color='lightcoral')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Cantidad Predicha (unidades)')
        plt.ylabel('Residuos')
        plt.title('Análisis de Residuos')
        plt.grid(True, alpha=0.3)
        graphs['residuos'] = self._plot_to_base64()
        
        # 3. Cantidad por región
        test_data = self.X_test.copy()
        test_data['cantidad_real'] = self.y_test
        if isinstance(test_data['region'].iloc[0], str):
            test_data['region_name'] = test_data['region']
        else:
            test_data['region_name'] = self.label_encoders['region'].inverse_transform(test_data['region'])
        
        region_data = test_data.groupby('region_name')['cantidad_real'].mean()
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(region_data.index, region_data.values, color=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title('Cantidad Promedio por Región')
        plt.ylabel('Cantidad (unidades)')
        plt.xticks(rotation=45)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        graphs['cantidad_region'] = self._plot_to_base64()
        
        # 4. Evolución temporal
        monthly_data = self.data_monthly.copy()
        monthly_data['año'] = monthly_data['año'].astype(int)
        monthly_data['mes'] = monthly_data['mes'].astype(int)
        monthly_data['fecha'] = pd.to_datetime(
            monthly_data['año'].astype(str) + '-' + monthly_data['mes'].astype(str).str.zfill(2) + '-01'
        )
        monthly_data = monthly_data.sort_values('fecha')
        
        if isinstance(monthly_data['region'].iloc[0], str):
            monthly_data['region_name'] = monthly_data['region']
        else:
            monthly_data['region_name'] = self.label_encoders['region'].inverse_transform(monthly_data['region'])
        
        plt.figure(figsize=(12, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, region in enumerate(monthly_data['region_name'].unique()):
            region_subset = monthly_data[monthly_data['region_name'] == region]
            plt.plot(region_subset['fecha'], region_subset['cantidad_total_mes'], 
                    marker='o', label=region, linewidth=2, color=colors[i % len(colors)])
        
        plt.xlabel('Fecha')
        plt.ylabel('Cantidad Total (unidades)')
        plt.title('Evolución Mensual de Cantidad por Región')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        graphs['evolucion_temporal'] = self._plot_to_base64()
        
        # 5. Distribución de errores
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='lightsteelblue')
        plt.xlabel('Error (unidades)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Errores')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)
        graphs['distribucion_errores'] = self._plot_to_base64()
        
        return graphs
    
    def _plot_to_base64(self):
        """Convierte el plot actual a base64"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    
    def run_full_analysis(self):
        """Ejecuta el análisis completo y retorna resultados"""
        try:
            # Extraer datos
            if not self.extract_bar_sales_data():
                return {"success": False, "error": "No se pudieron extraer datos"}
            
            # Crear características
            if not self.create_quantity_features():
                return {"success": False, "error": "Error al crear características"}
            
            # Entrenar modelo
            train_r2, test_r2, rmse, mae, cv_scores = self.train_linear_model()
            
            # Generar gráficos
            graphs = self.generate_graphs()
            
            # Predicciones para próximo mes
            next_month = datetime.now().month + 1
            if next_month > 12:
                next_month = 1
            predicciones = self.predict_quantity_by_region(next_month)
            
            # Métricas del modelo
            y_pred = self.model.predict(self.X_test_scaled)
            non_zero_mask = self.y_test != 0
            mape = np.mean(np.abs((self.y_test[non_zero_mask] - y_pred[non_zero_mask]) / self.y_test[non_zero_mask])) * 100
            
            return {
                "success": True,
                "graphs": graphs,
                "predicciones": predicciones,
                "metricas": {
                    "r2": round(test_r2, 4),
                    "rmse": round(rmse, 2),
                    "mae": round(mae, 2),
                    "mape": round(mape, 2)
                },
                "mes_prediccion": next_month,
                "año_prediccion": 2025
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}