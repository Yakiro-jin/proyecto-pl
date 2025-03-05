import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
import tkinter as tk
from tkinter import messagebox, ttk

# Función para resolver el problema primal (simplex)
def resolver_problema_lineal(modo, funcion_objetivo, restricciones, limites, operadores):
    try:
        # Convertir a numpy arrays
        c = np.array(funcion_objetivo, dtype=float)
        A = np.array(restricciones, dtype=float)
        b = np.array(limites, dtype=float)
        
        # Cambiar el signo de la función objetivo si es minimización
        if modo == "min":
            c = -c
        
        # Convertir los operadores a restricciones de desigualdad
        A_ub = []
        b_ub = []
        for i in range(len(operadores)):
            if operadores[i] == '<=':
                A_ub.append(A[i])
                b_ub.append(b[i])
            elif operadores[i] == '>=':
                A_ub.append(-A[i])
                b_ub.append(-b[i])
            else:
                raise ValueError("Operador no soportado. Use '<=' o '>='.")
        
        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)
        
        # Resolver el problema usando el método simplex
        resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, method='simplex')
        
        if resultado.success:
            resultado_texto = (
                f"Solución del problema primal (simplex):\n"
                f"Valor óptimo: {-resultado.fun if modo == 'min' else resultado.fun}\n"
                f"Variables de decisión: {resultado.x}"
            )
            messagebox.showinfo("Resultado Primal", resultado_texto)
        else:
            messagebox.showerror("Error", "No se encontró una solución óptima para el problema primal.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al resolver el problema primal: {str(e)}")

# Función para resolver el problema dual
def resolver_problema_dual(funcion_objetivo, restricciones, limites):
    try:
        # Definir las variables duales
        y = cp.Variable(len(limites))
        
        # Definir la función objetivo
        objetivo = cp.Maximize(cp.sum(cp.multiply(limites, y)))
        
        # Definir las restricciones
        restricciones_dual = [cp.sum(cp.multiply(restricciones[i], y)) <= funcion_objetivo[i] for i in range(len(funcion_objetivo))]
        
        # Resolver el problema dual
        problema_dual = cp.Problem(objetivo, restricciones_dual)
        problema_dual.solve()
        
        if problema_dual.status == cp.OPTIMAL:
            resultado_texto = (
                f"Solución del problema dual:\n"
                f"Valor óptimo: {problema_dual.value}\n"
                f"Variables duales: {y.value}"
            )
            messagebox.showinfo("Resultado Dual", resultado_texto)
        else:
            messagebox.showerror("Error", "No se encontró una solución óptima para el problema dual.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al resolver el problema dual: {str(e)}")

# Función para obtener los datos de la interfaz gráfica
def obtener_datos():
    try:
        # Obtener el modo
        modo = modo_var.get().strip().lower()
        if modo not in ['max', 'min']:
            raise ValueError("Modo no válido. Use 'max' o 'min'.")
        
        # Obtener la función objetivo
        funcion_objetivo = list(map(float, funcion_objetivo_entry.get().split()))
        
        # Obtener las restricciones
        restricciones = []
        for child in restricciones_frame.winfo_children():
            if isinstance(child, tk.Entry):
                coeficientes = list(map(float, child.get().split()))
                restricciones.append(coeficientes)
        
        # Obtener los límites
        limites = list(map(float, limites_entry.get().split()))
        
        # Obtener los operadores
        operadores = []
        for child in operadores_frame.winfo_children():
            if isinstance(child, ttk.Combobox):
                operadores.append(child.get())
        
        # Verificar que los datos sean consistentes
        if len(restricciones) != len(operadores) or len(restricciones[0]) != len(funcion_objetivo):
            raise ValueError("Los datos ingresados no son consistentes.")
        
        # Resolver el problema primal y dual
        resolver_problema_lineal(modo, funcion_objetivo, restricciones, limites, operadores)
        resolver_problema_dual(funcion_objetivo, restricciones, limites)
    except Exception as e:
        messagebox.showerror("Error", f"Error en los datos ingresados: {str(e)}")

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Resolución de Programación Lineal")

# Modo (max o min)
modo_label = tk.Label(root, text="Modo (max o min):")
modo_label.grid(row=0, column=0, padx=10, pady=10)
modo_var = tk.StringVar(value="max")
modo_entry = ttk.Combobox(root, textvariable=modo_var, values=["max", "min"])
modo_entry.grid(row=0, column=1, padx=10, pady=10)

# Función objetivo
funcion_objetivo_label = tk.Label(root, text="Función objetivo (coeficientes separados por espacios):")
funcion_objetivo_label.grid(row=1, column=0, padx=10, pady=10)
funcion_objetivo_entry = tk.Entry(root, width=40)
funcion_objetivo_entry.grid(row=1, column=1, padx=10, pady=10)

# Restricciones
restricciones_label = tk.Label(root, text="Restricciones (coeficientes separados por espacios):")
restricciones_label.grid(row=2, column=0, padx=10, pady=10)
restricciones_frame = tk.Frame(root)
restricciones_frame.grid(row=2, column=1, padx=10, pady=10)

# Límites
limites_label = tk.Label(root, text="Límites (separados por espacios):")
limites_label.grid(row=3, column=0, padx=10, pady=10)
limites_entry = tk.Entry(root, width=40)
limites_entry.grid(row=3, column=1, padx=10, pady=10)

# Operadores
operadores_label = tk.Label(root, text="Operadores (<= o >=):")
operadores_label.grid(row=4, column=0, padx=10, pady=10)
operadores_frame = tk.Frame(root)
operadores_frame.grid(row=4, column=1, padx=10, pady=10)

# Función para agregar restricciones dinámicamente
def agregar_restriccion():
    # Entrada para los coeficientes de la restricción
    restriccion_entry = tk.Entry(restricciones_frame, width=40)
    restriccion_entry.pack(padx=5, pady=5)
    
    # Combobox para el operador
    operador_combobox = ttk.Combobox(operadores_frame, values=["<=", ">="], width=5)
    operador_combobox.pack(padx=5, pady=5)
    operador_combobox.set("<=")

# Botón para agregar restricciones
agregar_restriccion_button = tk.Button(root, text="Agregar Restricción", command=agregar_restriccion)
agregar_restriccion_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Botón para resolver
resolver_button = tk.Button(root, text="Resolver", command=obtener_datos)
resolver_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

# Ejecutar la interfaz gráfica
root.mainloop()

# Ejemplo de uso
#modo = "max"
#funcion_objetivo = [32, 26, 20]
#restricciones = [[90, 20, 40], [30, 80, 60], [10, 20, 60]]
#limites = [200, 180, 150]
#operadores = ['<=', '<=', '<='] pip install scipy cvxpy instalar libreria para que funcione