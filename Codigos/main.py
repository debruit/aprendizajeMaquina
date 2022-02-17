
import sys
sys.path.insert( 0, '../../lib/python3' )


if __name__ == '__main__':
    #Regresión lineal
    linear=r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01\examples\python3\LinearModel_01.py"
    linearFIT = r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01\examples\python3\LinearModel_Fit_01.py"
    linearFITGra = r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01\examples\python3\LinearModel_Fit_GradientDescent_01.py"

    archivo=r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01_en_clase\examples\python3\data_00.csv"

    #regresión logistica
    logistic=r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01_en_clase\examples\python3\LogisticModel_01.py"
    readBPM=r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01_en_clase\examples\python3\read_pbm.py"

    imagen=r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01_en_clase\examples\python3\ejemplo.pbm"
    dataLogisitc=r"C:\Users\Guatavita\Documents\semestre 14\ML\Codigos\PUJ_ML_01_en_clase\examples\python3\prueba.csv"

    archivoExec=logistic
    sys.argv = [archivoExec, dataLogisitc]
    exec(open(archivoExec).read())
