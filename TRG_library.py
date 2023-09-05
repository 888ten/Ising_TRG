import numpy as np
import scipy.linalg as linalg
import pickle
from matplotlib import pyplot

# テンソルのトレースを計算する関数
def Trace_tensor(A):
    Trace_A = np.trace(A, axis1=0, axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A

# 初期テンソルを生成する関数
def initialize_A(T):
    A = np.empty((2, 2, 2, 2))

    for i in range(2):
        si = (i - 0.5) * 2
        for j in range(2):
            sj = (j - 0.5) * 2
            for k in range(2):
                sk = (k - 0.5) * 2
                for l in range(2):
                    sl = (l - 0.5) * 2

                    A[i, j, k, l] = np.exp((si * sj + sj * sk + sk * sl + sl * si) / T)

    factor = Trace_tensor(A)
    A /= factor
    return A, factor

# 特異値分解とテンソルの切り捨てを行う関数
def SVD_type1(A, D):
    A_mat1 = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2] * A.shape[3]))
    U, s, VT = linalg.svd(A_mat1)
    D_cut = min(s.size, D)
    s_t = np.sqrt(s[:D_cut])
    S3 = np.dot(U[:, :D_cut], np.diag(s_t))
    S1 = np.dot(np.diag(s_t), VT[:D_cut, :])
    S3 = np.reshape(S3, (A.shape[0], A.shape[1], D_cut))
    S1 = np.reshape(S1, (D_cut, A.shape[2], A.shape[3]))
    return S1, S3

# 特異値分解とテンソルの切り捨てを行う関数 (タイプ2)
def SVD_type2(A, D):
    A_mat2 = np.transpose(A, (0, 3, 1, 2))
    A_mat2 = np.reshape(A_mat2, (A.shape[0] * A.shape[3], A.shape[1] * A.shape[2]))
    U, s, VT = linalg.svd(A_mat2)
    D_cut = min(s.size, D)
    s_t = np.sqrt(s[:D_cut])
    S2 = np.dot(U[:, :D_cut], np.diag(s_t))
    S4 = np.dot(np.diag(s_t), VT[:D_cut, :])
    S2 = np.reshape(S2, (A.shape[0], A.shape[3], D_cut))
    S4 = np.reshape(S4, (D_cut, A.shape[1], A.shape[2]))
    return S2, S4

# テンソルの更新を行う関数
def Update_Atensor(A, D):
    S1, S3 = SVD_type1(A, D)
    S2, S4 = SVD_type2(A, D)
    A = Combine_four_S(S1, S2, S3, S4)
    factor = Trace_tensor(A)
    A /= factor
    return A, factor

# テンソルの結合を行う関数
def Combine_four_S(S1, S2, S3, S4):
    S12 = np.tensordot(S1, S2, axes=(1, 0))
    S43 = np.tensordot(S4, S3, axes=(2, 0))
    A = np.tensordot(S12, S43, axes=([1, 2], [1, 2]))
    A = np.transpose(A, (0, 1, 3, 2))
    return A

# TRG法によるIsingモデルの計算を行う関数
def TRG_Square_Ising(T, D, TRG_steps):
    A, factor = initialize_A(T)
    TRG_factors = [factor]
    
    for i_TRG in range(TRG_steps):
        A, factor = Update_Atensor(A, D)
        TRG_factors.append(factor)

    free_energy_density = 0.0
    for i_TRG in range(TRG_steps + 1):
        free_energy_density += np.log(TRG_factors[i_TRG]) * 0.5 ** i_TRG
    free_energy_density = -T * 0.5 * (free_energy_density + 0.5 ** TRG_steps * np.log(Trace_tensor(A)))

    return free_energy_density
