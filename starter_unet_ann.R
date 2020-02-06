#load libraries
library( ANTsR )
library( ANTsRNet )
library( keras )
# library(tensorflow)

#path to segmented reference images that comes with repo
path = '~/ml_fish_seg/fish_segmentations'

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
segmentationLabels <- c( 1, 2, 3 )
numberOfLabels <- length( segmentationLabels )
initialization <- paste0( 'KMeans[', numberOfLabels, ']' )
trainingBatchSize = length(reduced_imgs)

#define where to start (first image)
domainImage <- reduced_imgs[[1]]

#create empty arrays for training x/y data to fill?
X_train <- array( data = NA, dim = c( trainingBatchSize, dim( domainImage ), 3 ) )
Y_train <- array( data = NA, dim = c( trainingBatchSize, dim( reduced_segs[[1]] ) ) )


for( i in seq_len( trainingBatchSize ) )
    {
    cat( "Processing image", i, "\n" ) #gives visual of progress through images in training set
    X_train[i,,, 1:3] <- as.array( reduced_imgs[[i]] ) #populate with images
    Y_train[i,,] <- as.array( reduced_segs[[i]] ) #populate with segmentation
    }
  
  Y_train <- encodeUnet( Y_train, segmentationLabels ) #tags segmentations with segmentation labels (1-3)

  # Perform a simple normalization

  X_train <- ( X_train - mean( X_train ) ) / sd( X_train )


model <- createUnetModel2D( c( dim( domainImage ), 1 ),
    numberOfOutputs = numberOfLabels ) #create model with first image in test set

model %>% compile( loss = loss_categorical_crossentropy,
    optimizer = optimizer_adam( lr = 0.0001 )  ) #configures a Keras model for training

track <- model %>% fit( X_train, Y_train,
           epochs = 100, batch_size = 4, verbose = 1, shuffle = TRUE,
           validation_split = 0.2 ) #Trains the model for a fixed number of epochs (iterations on a dataset).
          #This last bit is currently struggling bc the dimentions of the image (x) and segmentation (y) arrays have different dimentions 
          #work on fixing this tomorrow when not brain dead - also save this whole thing as a rmd tomorrow here and in my helens drive

#at this point the model has been trained but not tested on any new data
