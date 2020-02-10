#load libraries
library( ANTsR )
library( ANTsRNet )
library( keras )
# library(tensorflow)

predicted2segmentation <- function( x, domainImage ) {
  xdim = dim( x )
  nclasses = tail( xdim, 1 )
  nvoxels = prod( head( xdim, domainImage@dimension ) )
  pmat = matrix( nrow = nclasses, ncol = nvoxels )
  for ( j in 1:nclasses ) pmat[j,] = x[,,j]
  segvec = apply( pmat, MARGIN=2, FUN=which.max )
  seg = makeImage( head( xdim, domainImage@dimension ), segvec )
  return( seg )
}

#path to segmented reference images that comes with repo
path = './fish_segmentations/'

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


for( i in seq_len( trainingBatchSize ) ) {
    cat( "Processing image", i, "\n" ) #gives visual of progress through images in training set
    splitter = splitChannels( reduced_imgs[[i]] )
    for ( j in 1:3 ) { # channels
      X[i,,, j] <- as.array( splitter[[j]] ) #populate with images
      }
    Y[i,,] <- as.array( reduced_segs[[i]] ) #populate with segmentation
    }

Y <- encodeUnet( Y, segmentationLabels ) #tags segmentations with segmentation labels (1-3)

# verify the encoding etc is correct
whichTestImage = 5
testimg = as.antsImage( X[whichTestImage,,,1] )
segprob  = as.antsImage( Y[whichTestImage,,,2 ] )
plot( testimg, segprob, doCropping = FALSE, window.overlay=c(0.2,1) )
seg = predicted2segmentation( Y[whichTestImage,,,], domainImage )
plot( testimg, seg, doCropping = FALSE, window.overlay=c(2,5) )



train_indices = sample(1:nrow(X), round( 0.9 * nrow(X) ) )
X_train = X[train_indices,,,]
Y_train = Y[train_indices,,,]
X_test = X[-train_indices,,,]
Y_test = Y[-train_indices,,,]

model <- createUnetModel2D( c( dim( domainImage ), 3 ),
  numberOfOutputs = 4 , mode = 'classification' ) # create model with first image in test set

model %>% compile( loss = loss_categorical_crossentropy,
    optimizer = optimizer_adam( lr = 0.0001 )  ) #configures a Keras model for training

track <- model %>% fit( X_train, Y_train,
  epochs = 200, batch_size = 4, verbose = 1, shuffle = TRUE)
# Trains the model for a fixed number of epochs (iterations on a dataset).


#at this point the model has been trained but not tested on any new data
predicted <- predict( model, X_test )
whichTestImage = 2
testimg = as.antsImage( X_test[whichTestImage,,,1] )
seg = predicted2segmentation( Y_test[whichTestImage,,,], domainImage )
# better approach:
#  resample probability images to full resolution, then refine, then
#  derive segmentation from probability images - can refine with atropos
#  or other tools
plot( testimg, seg, doCropping = FALSE, window.overlay=c(2,5), alpha=.5 )

#testing the unseen 50 images with manual segmentation

#If you want to continue from the existing model. 
#load('./.RData')
unseen = dir(patt='jpg', path = '50segmentations/', full.names = T)

#if we want to calculate a dice coefficient for accuracy or something.
#unseen.seg = dir(patt='gz', path = '50segmentations/', full.names = T)

new.imgs = manual= list()

for (i in 1:length(unseen)) new.imgs[[i]] = antsImageRead(unseen[i]) %>% 
              resampleImage(., c(256,256), useVoxels = T)

#this is causing a strange error. I can't plot the original images with the segmentations either. 

#for (i in 1:length(unseen.seg)) manual[[i]] = antsImageRead(unseen.seg[i]) %>% resampleImage(., c(256,256), useVoxels = T, interpType = 'nearestNeighbor')

new.X <- array( data = NA, dim = c( length(unseen), dim( domainImage ), 3 ) )

for( i in 1:length( unseen ) ) {
  cat( "Processing image", i, "\n" ) #gives visual of progress through images in training set
  splitter = splitChannels( new.imgs[[i]] )
  for ( j in 1:3 ) { # channels
    new.X[i,,, j] <- as.array( splitter[[j]] ) #populate with images
  }
}


predicted <- predict( model, new.X )
#for (i in 1:length(unseen)) whichTestImage = 1
testimg = as.antsImage( new.X[whichTestImage,,,1] )
seg = predicted2segmentation( predicted[whichTestImage,,,], domainImage )
# better approach:
#  resample probability images to full resolution, then refine, then
#  derive segmentation from probability images - can refine with atropos
#  or other tools
plot( testimg, seg, doCropping = FALSE, window.overlay=c(2,5), alpha=.5 )

antsImageWrite(seg, filename = paste(unseen.seg[whichTestImage], "_Pred.nii.gz", sep=''))
