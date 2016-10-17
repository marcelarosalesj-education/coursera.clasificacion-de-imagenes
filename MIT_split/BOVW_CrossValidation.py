from BOVW_functions import *

##########################
#PAR�METROS DE EJECUCION
##########################

# M�todo usado para detectar los puntos de inter�s
detector='SIFT'
# M�todo usado para obtener la descripci�n de los puntos de inter�s
descriptor='SIFT'
# N�mero de puntos de inter�s que se utilizan para obtener el vocabulario visual
num_samples=50000
# N�mero de palabras en el vocabulario visual
k=32
# Par�metros para realizar la validaci�n cruzada en el aprendizaje del clasificaro
# folds: n� de particioines
# start, end: valor inicial y final del factor de regularizaci�n C que se validar�n
# mumparams: n� de valores diferentes para el factor de regularizaci�n C entre start y end que se van a validar
folds=5
start=0.01
end=10
numparams=30
# Directorio raiz donde se encuentran todas las im�genes de aprendizaje
dataset_folder_train='../../Databases/MIT_split/train/'
# Directorio raiz donde se encuentran todas las im�genes de test
dataset_folder_test='../../Databases/MIT_split/test/'

##############################################

# Preparaci�n de los nombres de los ficheros necesarios para guardar el vocabulario y las palabras visuales de las im�genes de aprendizaje y test
codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

# C�lculo de puntos de inter�s para todas las im�genes del conjunto de aprendizaje
filenames_train,GT_ids_train,GT_labels_train = prepareFiles(dataset_folder_train)
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)

# Construcci�n del vocabulario visual. El vocabulario queda guardado en disco.
# Comentar esta l�nea si el vocabulario ya est� creado y guardado en disco de una ejecuci�n anterior
CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)

# Carga de un vocabulario visual previamente creado y guardado en disco en una ejecuci�n anterior.
# Comentar esta l�nea si se quiere re-calcular el vocabulario o si el vocabulario todav�a no se ha creado
#CB=cPickle.load(open(codebook_filename,'r'))

# Obtiene la descripci�n BoW de las im�genes del conjunto de aprendizaje
VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)

# Carga de la descripci�n BoW del conjunto de aprendizaje previamente creado y guardado en disco en una ejecuci�n anterior.
# Comentar esta l�nea si se quiere re-calcular la representaci�n o si la representaci�n todav�a no se ha creado
#VW_train=cPickle.load(open(visual_words_filename_train,'r'))

# C�lculo de puntos de inter�s para todas las im�genes del conjunto de test
filenames_test,GT_ids_test,GT_labels_test = prepareFiles(dataset_folder_test)
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)

# Obtiene la descripci�n BoW de las im�genes del conjunto de test
VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)

# Entrena un clasificador SVM con las im�genes del conjunto de aprendizaje y lo eval�a utilizando las im�genes del conjunto de test
# El valor del factor de regularizaci�n C se obtiene por validaci�n cruzada
# Devuelve la accuracy como medida del rendimiento del clasificador
ac_BOVW_L = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

print 'Accuracy BOVW: '+str(ac_BOVW_L)

