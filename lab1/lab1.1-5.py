import numpy as np

# ZADANIE 1
def kwadraty(input_list):
    output_list = [number**2 for number in input_list if number >= 0]
    return output_list

# ZADANIE 2
print(np.arange(1,51).reshape(5, 10))

# ZADANIE 3
def wlasciwosci_macierzy(A):
    liczba_elementow =  A.size
    liczba_kolumn =  A.shape[1]
    liczba_wierszy =  A.shape[0]
    srednie_wg_wierszy = A.mean(axis=1)
    srednie_wg_kolumn = A.mean(axis=0)
    kolumna_2 = A[:,2]
    wiersz_3 =  A[3,:]
    return (
        liczba_elementow, liczba_kolumn, liczba_wierszy, 
        srednie_wg_wierszy, srednie_wg_kolumn,
        kolumna_2, wiersz_3)

# ZADANIE 4
def dzialanie1(A, x):
    """ iloczyn macierzy A z wektorem x """
    return A.dot(x)

def dzialanie2(A, B):
    """ iloczyn macierzy A · B """
    return np.dot(A,B)

def dzialanie3(A):
    """ odwrotność iloczynu A · A.T """
    return np.linalg.inv(np.dot(A, A.transpose()))

def dzialanie4(A, B):
    """ wynik działania (A · B)^T - B^T · A^T """
    return np.dot(A,B).transpose() - np.dot(B.transpose(),A.transpose())

