from BOVW_functions import *

##########################
#PARaMETROS DE EJECUCION
##########################

# Metodo usado para detectar los puntos de interes
detector='SIFT'
# Metodo usado para obtener la descripcion de los puntos de interes
descriptor='SIFT'
# Numero de puntos de interes que se utilizan para obtener el vocabulario visual
num_samples=50000
# Numero de palabras en el vocabulario visual
k=32
# Paremetros para realizar la validacion cruzada en el aprendizaje del clasificaro
# folds: nº de particioines
# start, end: valor inicial y final del factor de regularizacion C que se validaran
# mumparams: nº de valores diferentes para el factor de regularizacion C entre start y end que se van a validar
folds=5
start=0.01
end=10
numparams=30
# Directorio raiz donde se encuentran todas las imagenes de aprendizaje
dataset_folder_train='../../Databases/MIT_split/train/'
# Directorio raiz donde se encuentran todas las imagenes de test
dataset_folder_test='../../Databases/MIT_split/test/'

##############################################

# Preparacion de los nombres de los ficheros necesarios para guardar el vocabulario y las palabras visuales de las imagenes de aprendizaje y test
codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

# Calculo de puntos de interes para todas las imagenes del conjunto de aprendizaje
filenames_train,GT_ids_train,GT_labels_train = prepareFiles(dataset_folder_train)
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)

# Construccion del vocabulario visual. El vocabulario queda guardado en disco.
# Comentar esta linea si el vocabulario ya esta creado y guardado en disco de una ejecucion anterior
CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)

# Carga de un vocabulario visual previamente creado y guardado en disco en una ejecucion anterior.
# Comentar esta linea si se quiere re-calcular el vocabulario o si el vocabulario todavia no se ha creado
#CB=cPickle.load(open(codebook_filename,'r'))

# Obtiene la descripcion BoW de las imagenes del conjunto de aprendizaje
VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)

# Carga de la descripcion BoW del conjunto de aprendizaje previamente creado y guardado en disco en una ejecucion anterior.
# Comentar esta linea si se quiere re-calcular la representacion o si la representacion todavia no se ha creado
#VW_train=cPickle.load(open(visual_words_filename_train,'r'))

# Calculo de puntos de interes para todas las imagenes del conjunto de test
filenames_test,GT_ids_test,GT_labels_test = prepareFiles(dataset_folder_test)
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)

# Obtiene la descripcion BoW de las imagenes del conjunto de test
VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)

# Entrena un clasificador SVM con las imagenes del conjunto de aprendizaje y lo evalua utilizando las imagenes del conjunto de test
# El valor del factor de regularizacion C se obtiene por validacion cruzada
# Devuelve la accuracy como medida del rendimiento del clasificador
ac_BOVW_L = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

print 'Accuracy BOVW: '+str(ac_BOVW_L)

