import numpy as np

# Escalonar matriz para triangular superior
def escalate_to_upper_triangular(a: np.ndarray, b: np.ndarray):
    n = len(b)
    c = np.zeros(n)
    for i in range(0, n):
        for j in range(i+1, n):
            if a[j][i] != 0:
                factor = a[j][i] / a[i][i]
                c[i][j] = -factor
                for k in range(i, n):
                    a[j][k] -= factor * a[i][k]
                b[j] -= factor * b[i]
    return a, b, c

# Resolver matrix triangular superior
def solve_upper_triangular(a: np.ndarray, b: np.ndarray):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]
    return x

# Resolver por metodo de eliminação de Gauss
def gauss_elimination(a: np.ndarray, b: np.ndarray):
    a, b, c = escalate_to_upper_triangular(a, b)
    del(c)  # c não sera usado
    return solve_upper_triangular(a, b)

# Resolver por metodo de eliminação de Gauss com pivotação parcial
def partial_pivot(a: np.ndarray, b: np.ndarray):
    max = a.argmax(axis=0)[0]
    for i in range(len(a)):
        # Swap rows in a matrix
        temp = a[i].copy()
        a[i] = a[max]
        a[max] = temp

        # Swap corresponding values in b vector
        temp = b[i].copy()
        b[i] = b[max]
        b[max] = temp
    
    return gauss_elimination(a, b)

# Resolver por metodo de decomposição LU
def LU_decomposition(a: np.ndarray, b: np.ndarray):
    a, b, c = escalate_to_upper_triangular(a, b)

# Resolver por metodo de decomposição LU com pivotação parcial
def LU_partial_pivot(a: np.ndarray, b: np.ndarray):
    raise NotImplementedError("Function not implemented yet.")

# Exemplo de uso
if __name__ == "__main__":
    # Menu de seleção
    print("This module provides functions for solving linear systems using various methods.")
    print("available functions:")
    print("[1] Gauss Elimination")
    print("[2] Gauss Elimination with Partial Pivoting")
    print("[3] LU Decomposition")
    print("[4] LU Decomposition with Partial Pivoting")
    choice = input("Choose a function (1-4): ")
    
    while True:
        try:
            if choice in ["1", "2", "3", "4"]:
                break
            else:
                choice = input("Invalid choice. Please choose a function (1-4): ")
        except ValueError:
            choice = input("Invalid input. Please choose a function (1-4): ")

    while choice not in ["1", "2", "3", "4"]:
        choice = input("Invalid choice. Please choose a function (1-4): ")

    # Entrada da matrix A
    n = int(input("Enter the order of the matrix: "))
    print("Enter the matrix a (row by row, separated by spaces):")
    a = []
    for _ in range(n):
        row = list(map(float, input().split()))
        while len(row) != n:
            print(f"Row must have {n} elements. Please re-enter the row:")
            row = list(map(float, input().split()))
        a.append(row)
    a = np.array(a, dtype=float)

    # Entrada do vetor b
    b = []
    print("Enter the vector b (separated by spaces):")
    b = np.array(list(map(float, input().split())))
    while len(b) != n:
        print(f"vector must have {n} elements. Please re-enter the vector:")
        b = np.array(list(map(float, input().split())))
    
    # Executar a função escolhida
    result = None
    if choice == "1":
        print("Calling gauss_elimination(a, b)...")
        result = gauss_elimination(a, b)
    elif choice == "2":
        print("Calling partial_pivot(a, b, 0)...")
        result = partial_pivot(a, b)
    elif choice == "3":
        print("Calling LU_decomposition(a)...")
        result = LU_decomposition(a, b)
    elif choice == "4":
        print("Calling LU_partial_pivot(a)...")
        result = LU_partial_pivot(a, b)
    else:
        print("Invalid choice.")

    print("Result:", result)