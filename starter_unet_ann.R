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
modelfn = './models/starter_unet.h5'
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

nEpochs = 200
if ( file.exists( modelfn ) ) {
  model = load_model_hdf5( modelfn )
  nEpochs = 20 # fine tune
  }
model %>% compile( loss = loss_categorical_crossentropy,
    optimizer = optimizer_adam( lr = 0.0001 )  ) #configures a Keras model for training
if ( nEpochs > 0 )
  track <- model %>% fit( X_train, Y_train,
    epochs = nEpochs, batch_size = 4, verbose = 1, shuffle = TRUE)
# Trains the model for a fixed number of epochs (iterations on a dataset).
# save_model_hdf5( model, modelfn )

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
unseen.seg = dir(patt='gz', path = '50segmentations/', full.names = T)

new.imgs = manual= list()
for (i in 1:length(unseen)) {
  new.imgs[[i]] = antsImageRead(unseen[i])
  splitter = splitChannels(  new.imgs[[i]] )
  manual[[i]] = antsImageRead(unseen.seg[i],dimension=2) %>% decropImage(  splitter[[1]])
  manual[[i]] = resampleImage( manual[[i]], c(256,256), useVoxels = T, interpType = "nearestneighbor")
  manual[[i]] = manual[[i]] * thresholdImage( manual[[i]], 1, 5 ) # decropping issue
  new.imgs[[i]] = resampleImage( new.imgs[[i]], c(256,256), useVoxels = T, interpType = "nearestneighbor")
}


new.X <- array( data = NA, dim = c( length(unseen), dim( domainImage ), 3 ) )
new.Y <- array( data = NA, dim = c( length(unseen), dim( manual[[1]] ) ) )
for( i in 1:length( unseen ) ) {
  cat( "Processing image", i, "\n" ) #gives visual of progress through images in training set
  splitter = splitChannels( new.imgs[[i]] )
  for ( j in 1:3 ) { # channels
    new.X[i,,, j] <- as.array( splitter[[j]] ) #populate with images
    new.Y[i,,] <- as.array( manual[[i]] )
  }
}
new.Y <- encodeUnet( new.Y, segmentationLabels ) #tags segmentations with segmentation labels (1-3)


train_indices = sample(1:nrow(new.X), round( 0.9 * nrow(new.X) ) )
X_train = new.X[train_indices,,,]
Y_train = new.Y[train_indices,,,]
X_test = new.X[-train_indices,,,]
Y_test = new.Y[-train_indices,,,]


nEpochs = 200
# start fresh - data may not be harmonized enough
# solution is to either harmonize or sample from both sets in training
# probably worth just harmonizing either code or data to sample from both sets
# or perhaps just abandon the first set of labels .... not as good. 
# also, looks like some post-processing will help when the fish / ruler / label
# arrangement is differently configured.
# a more balanced sampling (with augmentation) would also resolve this.
#
#
# create a generator with scaling shearing
randAff <- function( loctx,  txtype = "ScaleShear", sdAffine) {
  idim = 2
  noisemat = stats::rnorm(length(idparams), mean = 0, sd = sdAffine)
  if (txtype == "Translation")
    noisemat[1:(length(idparams) - idim )] = 0
  idparams = idparams + noisemat
  idmat = matrix(idparams[1:(length(idparams) - idim )],
              ncol = idim )
  idmat = polarX(idmat)
          if (txtype == "Rigid")
              idmat = idmat$Z
          if (txtype == "Affine")
              idmat = idmat$Xtilde
          if (txtype == "ScaleShear")
              idmat = idmat$P
          idparams[1:(length(idparams) - idim )] = as.numeric(idmat)
  setAntsrTransformParameters(loctx, idparams)
  return(loctx)
  }
polarX <- function(X) {
        x_svd <- svd(X)
        P <- x_svd$u %*% diag(x_svd$d) %*% t(x_svd$u)
        Z <- x_svd$u %*% t(x_svd$v)
        if (det(Z) < 0)
            Z = Z * (-1)
        return(list(P = P, Z = Z, Xtilde = P %*% Z))
    }

fixedParams = getCenterOfMass( manual[[1]] * 0 + 1 ) + rnorm( 2, 0, 1 )
loctx <- createAntsrTransform(precision = "float", type = "AffineTransform",
        dimension = 2 )
setAntsrTransformFixedParameters(loctx, fixedParams)
idparams = getAntsrTransformParameters(loctx)
shapeSD = 0.05
txType='ScaleShear'
loctx = randAff( loctx, sdAffine=shapeSD, txtype = txType )
augged = applyAntsrTransformToImage( loctx, new.imgs[[1]], new.imgs[[1]], interpolation = "nearestNeighbor")
plot( splitChannels(new.imgs[[1]])[[1]], splitChannels(augged)[[1]], alpha=0.5 )
# the above code can be put into a generator function and then 
# applied with fit_generator in order to make the training more robust.
# the alternative is to just train over augmented X for a given # of epochs
# then re-augment and continue on.

model2 <- createUnetModel2D( c( dim( domainImage ), 3 ),
   numberOfOutputs = 4 , mode = 'classification' ) # create model with first image in test set

model2 %>% compile( loss = loss_categorical_crossentropy,
    optimizer = optimizer_adam( lr = 0.0001 )  ) #configures a Keras model for training
track <- model2 %>% fit( X_train, Y_train,
  epochs = nEpochs, batch_size = 4, verbose = 1, shuffle = TRUE)


predicted <- predict( model2, X_test )
for ( whichTestImage in 1:nrow( X_test ) ) {
  print( whichTestImage )
  testimg = as.antsImage( X_test[whichTestImage,,,1] )
  realseg = as.antsImage( Y_test[whichTestImage,,,2] )
  testseg = as.antsImage( predicted[whichTestImage,,,2] )
  seg = predicted2segmentation( predicted[whichTestImage,,,], testimg )
  plot( testimg, seg, doCropping = FALSE, window.overlay=c(2,5), alpha=.5 )
  Sys.sleep( 5 )
}
save_model_hdf5( model2, modelfn )
# load_model_hdf5( modelfn ) will restore the model
# antsImageWrite(seg, filename = paste(unseen.seg[whichTestImage], "_Pred.nii.gz", sep=''))
