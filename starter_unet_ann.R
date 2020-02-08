#load libraries
library( ANTsR )
library( ANTsRNet )
library( keras )
# library(tensorflow)

#path to segmented reference images that comes with repo
path = './fish_segmentations'

seg_files = dir(patt='nii', path = path, full.names = T) #segmentation files end in nii
img_files = dir(patt='jpg', path = path, full.names = T) #original RGB image files end in jpg


#create empty lists for images/ segmentations to populate?
reduced_imgs = images <- list()
reduced_segs = segmentations <- list()

for (i in 1:length(seg_files)) segmentations [[i]] = antsImageRead(seg_files[i]) #read in segmentations from files
for (i in 1:length(img_files)) images [[i]] = antsImageRead(img_files[i]) #read in images from file

#resample segmentations/ images to decrease file sizes/ make process run faster
for (i in 1:length(seg_files)) reduced_segs [[i]] = resampleImage(segmentations[[i]], c(256, 256), useVoxels = TRUE, interpType = 'nearestNeighbor')
for (i in 1:length(img_files)) reduced_imgs [[i]] = resampleImage(images[[i]], c(256, 256), useVoxels = TRUE, interpType = 'linear')

#define segments
segmentationLabels <- c( 0, 1, 2, 3 )
numberOfLabels <- length( segmentationLabels )
initialization <- paste0( 'KMeans[', numberOfLabels, ']' )
trainingBatchSize = length(reduced_imgs)

#define where to start (first image)
domainImage <- reduced_imgs[[1]]

#create empty arrays for training x/y data to fill?
X <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ), 3 ) )
Y <- array( data = NA, dim = c( trainingBatchSize, dim( reduced_segs[[1]] ) ) )


for( i in seq_len( trainingBatchSize ) )
    {
    cat( "Processing image", i, "\n" ) #gives visual of progress through images in training set
    X[i,,, 1:3] <- as.array( reduced_imgs[[i]] ) #populate with images
    Y[i,,] <- as.array( reduced_segs[[i]] ) #populate with segmentation
    }

  Y <- encodeUnet( Y, segmentationLabels ) #tags segmentations with segmentation labels (1-3)

  # Perform a simple normalization

  X <- ( X - mean( X ) ) / sd( X )
train_indices = 1:14
X_train = X[train_indices,,,]
Y_train = Y[train_indices,,,]
X_test = X[-train_indices,,,]
Y_test = Y[-train_indices,,,]

model <- createUnetModel2D( c( dim( domainImage ), 3 ),
    numberOfOutputs = 4 , mode = 'classification' ) #create model with first image in test set

model %>% compile( loss = loss_categorical_crossentropy,
    optimizer = optimizer_adam( lr = 0.0001 )  ) #configures a Keras model for training

track <- model %>% fit( X_train, Y_train,
  epochs = 10, batch_size = 4, verbose = 1, shuffle = TRUE)
# Trains the model for a fixed number of epochs (iterations on a dataset).

#at this point the model has been trained but not tested on any new data
predicted <- predict( model, X_test )

predicted2segmentation <- function( x, domainImage ) {
  xdim = dim( x )
  nclasses = tail( xdim, 1 )
  nvoxels = prod( head( xdim, domainImage@dimension ) )
  pmat = matrix( x, nrow  = nclasses, ncol = nvoxels )
  segvec = apply( pmat, MARGIN=2, FUN=which.max )
  makeImage( head( xdim, domainImage@dimension ), segvec )
}

seg = predicted2segmentation( predicted[1,,,], domainImage )
