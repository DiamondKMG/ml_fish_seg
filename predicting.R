#load libraries
library( ANTsR )
library( ANTsRNet )
library( tensorflow )
library( keras )


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

f=dir(patt='IN', path='/home/stnava/INHS/256px/', full.names =T )
g=dir(patt='IN', path='/home/stnava/INHS/', full.names =T )
file.names = dir(patt='IN', path='/home/stnava/INHS/256px/', full.names =F )

N=30
#samples=1:N
samples=sample(1:length(f), N)
f=f[samples]
g=g[samples]
file.names=file.names[samples]
  
imglist = lapply(X = f, FUN = antsImageRead)
full.imgs = lapply(X= g, FUN = antsImageRead)
modelfn = '/home/maga/ml_fish_seg/models/augmented_unet.h5'
domainImage <- imglist[[1]]

new.X <- array( data = NA, dim = c( length(f), dim( domainImage ), 3 ) )
for( i in 1:length( f ) ) {
  cat( "Processing image", i, "\n" ) #gives visual of progress through images in training set
  splitter = splitChannels( imglist[[i]] )
  for ( j in 1:3 ) { # channels
    new.X[i,,, j] <- as.array( splitter[[j]] ) #populate with images
  }
}


model2 <- createUnetModel2D( c( dim( domainImage ), 3 ),
                             numberOfOutputs = 4 , mode = 'classification' ) # create model with first image in test set

if ( file.exists( modelfn ) ) {
  load_model_weights_hdf5( model2, modelfn )
  nEpochs = 5 # fine tune
}

model2 %>% compile( loss = keras::loss_categorical_crossentropy,
                    optimizer = optimizer_adam( lr = 0.0001 )  ) #configures a Keras model for training

predicted <- predict( model2, new.X )

for ( whichTestImage in 1:length(f)) {
  print( f[whichTestImage] )
  testimg = as.antsImage( new.X[whichTestImage,,,1])
  seg = predicted2segmentation( predicted[whichTestImage,,,], testimg )
  #antsSetSpacing(seg, antsGetSpacing(imglist[[i]]))
  thresholded = thresholdImage(seg, 2, 2)
  thresholded = iMath(thresholded, 'GetLargestComponent')
  thresholded = iMath(thresholded, "MD", 10 )
  

  thresholded = resampleImage(thresholded, dim(full.imgs[[whichTestImage]]), useVoxels = T, interpType = 'nearestNeighbor')
  antsSetSpacing(thresholded, antsGetSpacing(full.imgs[[whichTestImage]]))
  
  temp = splitChannels(full.imgs[[whichTestImage]])
  masked = lapply(X=temp, FUN=maskImage, thresholded )
  cropped = lapply(X=masked, FUN=cropImage, thresholded)
  merged = mergeChannels(cropped)
  antsImageWrite(antsImageClone(merged, 'unsigned char'), paste('/scratch/', file.names[whichTestImage],sep=''))
  
  #plot( testimg, seg, doCropping = FALSE, window.overlay=c(2,5), alpha=.5 )
  #Sys.sleep( 2 )
  #plot( testimg, thresholded, doCropping = FALSE,  alpha=.5 )
  #Sys.sleep( 2 )
  #masked = maskImage(imglist[[i]], thresholded)
  #cropped = cropImage(masked, thresholded)
  #plot(cropped)
}


