import numpy as np
from PIL import Image
import math
import time
import os
start_time = time.time()

def DisplayImage(presenting_image):
    python = np.array(presenting_image)
    python = Image.fromarray(python)
    python.show()
    #with np.printoptions(threshold=np.inf):
    #    print(python)
##################################################
# Image Loading & Cropping
##################################################
# parameter setting
ImageSize = 256
small_filter_size = 7
large_filter_size = 41
Gaussian_sigma = 1

# Manual coefficient
x_co = 230
y_co = 250

def ImageOpening(image_name,x_coor, y_coor):
    return Image.open('./'+image_name).convert('L').crop((x_coor, y_coor, x_coor + ImageSize, y_coor + ImageSize))
image = ImageOpening('lena.jpg', 230, 250)
#image = ImageOpening('lena.jpg', 0, 0)
# example::: image = Image.open('./ChunSewhan.jpg').convert('L').crop((x_co, y_co, x_co + ImageSize, y_co + ImageSize))

##################################################
# Image Instance Setting
##################################################

## Image Lists: Initialization

# A_1: Image for spatial domain_without padding
# A_2: Image for spatial domain_with small padding
# A_3: Image for spatial domain_with large padding

# B_1: Image for frequency domain_without padding
# B_2: Image for frequency domain_with small padding
# B_2: Image for frequency domain_with large padding

def ImageInitialization(ReferenceImage, FilterSize=1):
    return Image.new('L', (ReferenceImage + FilterSize - 1, ReferenceImage + FilterSize - 1))

A_1 = ImageInitialization(image.size[0])
A_2 = ImageInitialization(image.size[0], small_filter_size)
A_3 = ImageInitialization(image.size[0], large_filter_size)
B_1 = ImageInitialization(image.size[0])
B_2 = ImageInitialization(image.size[0], small_filter_size)
B_3 = ImageInitialization(image.size[0], large_filter_size)

#A_2 and A_3 have their image in the center for "uniform-side padding"
A_1.paste(image)
A_2.paste(image, box=(int(float(small_filter_size / 2) - 0.5), int(float(small_filter_size / 2) - 0.5)))
A_3.paste(image, box=(int(float(large_filter_size / 2) - 0.5), int(float(large_filter_size / 2) - 0.5)))
B_1.paste(image)
B_2.paste(image)
B_3.paste(image)

##################################################
# Filter Setting
##################################################

##Filter List: Initialization
    #for spatial domain:
# Filter_1: large(41x41) size filter_average filter (k=41)
# Filter_2: small(7x7) size filter_average filter (k=7)
# Filter_3: large(41x41) size filter_Gaussian filter (k=41)
# Filter_4: small(7x7) size filter_Gaussian filter (k=7)

# Gaussian Calculation(approximation=value dependent, rounding to 3rd than to represent minimum value as 1)
def GaussianFilterCalculation(Filter, sigmaValue):
    outputFilter = np.empty([Filter.shape[0], Filter.shape[1]])
    outputFilter_2 = np.empty([Filter.shape[0], Filter.shape[1]])
    for y in range(0, Filter.shape[0]):
        for x in range(0, Filter.shape[1]):
            Filter[y][x] = math.exp(
                (((x - ((Filter.shape[1] - 1) / 2)) ** 2) + ((y - ((Filter.shape[0] - 1) / 2)) ** 2))
                / ((-2) * (sigmaValue ** 2))) / (2 * math.pi * (sigmaValue ** 2))
    for y in range(0, Filter.shape[0]):
        for x in range(0, Filter.shape[1]):
            outputFilter_2[y][x] = round(Filter[y][x], 10)
    for y in range(0, Filter.shape[0]):
        for x in range(0, Filter.shape[1]):
            outputFilter[y][x] = int(round(outputFilter_2[y][x] / np.min(outputFilter_2[np.nonzero(outputFilter_2)])))
    # Sum of the Filter should be 1
    outputFilter = outputFilter / outputFilter.sum()
    return outputFilter

# Filter Initialization
Filter_1 = np.ones([large_filter_size, large_filter_size]) / (large_filter_size ** 2)
Filter_2 = np.ones([small_filter_size, small_filter_size]) / (small_filter_size ** 2)
Filter_3 = GaussianFilterCalculation(np.empty([large_filter_size, large_filter_size]), Gaussian_sigma)
Filter_4 = GaussianFilterCalculation(np.empty([small_filter_size, small_filter_size]), Gaussian_sigma)



##################################################
# Spatial Convolution Filtering
##################################################

def Spatial_Convolution_Filtering(img_input, filter_input):
    #img_input format: PIL.image format
    #filter_input format: numpy array
    img_output = Image.new('L', (
        img_input.size[0] - filter_input.shape[1] + 1, img_input.size[0] - filter_input.shape[0] + 1))
    input_pixel = img_input.load()
    output_pixel = img_output.load()
    AddingBlock = np.empty((filter_input.shape[0], filter_input.shape[1]))
    print(" Input Image Size: ", img_input.size, "\n", "Output Image Size: ", img_output.size)
    print("Loading...")
    for y in range(0, img_output.size[0]):
        for x in range(0, img_output.size[1]):

            for y2 in range(0, filter_input.shape[0]):
                for x2 in range(0, filter_input.shape[1]):
                    AddingBlock[y2][x2] = filter_input[y2][x2] * \
                                          (input_pixel[(y + y2), (x + x2)])
            output_pixel[y, x] = int(AddingBlock.sum())
        print(str(y) + '/' + str(img_output.size[0]))
    return img_output

##################################################
# frequency Filtering
##################################################

def Frequency_Domain_Multiplication_Filtering(img_input, filter_input):
    Display = True
    De_padding = False
    centering = False
    #img_input format: PIL.image format
    #filter_input format: numpy array

    #Local Methods
    def Display_in_frequency_domain(numpy_array_input):
        numpy_array = np.array(numpy_array_input)

        #Centralization
        for a in range(0, numpy_array.shape[1]):
            for b in range(0, numpy_array.shape[0]):
                numpy_array[b, a] = numpy_array[b, a] * ((-1) ** (b + a))
        numpy_array= np.fft.fft2(numpy_array)
        numpy_array_magnitude = np.empty_like(numpy_array)
        for a in range(0, numpy_array.shape[0]):
            for b in range(0, numpy_array.shape[1]):
                numpy_array_magnitude[a, b] = round(
                    math.sqrt(((numpy_array[a, b].real) ** 2) + ((numpy_array[a, b].imag) ** 2)), 10)
        #Change the BrightnessParameter to adjust the brightness of the freq. domain image. 1~255
        BrightnessParameter = 128
        numpy_array_magnitude = numpy_array_magnitude*BrightnessParameter/numpy_array_magnitude.mean()
        '''/numpy_array_magnitude.max()'''
        numpy_array_magnitude = numpy_array_magnitude.real.astype(int)
        Image.fromarray(numpy_array_magnitude).show()
        return

        # fourier transformation
        '''
        # def FT_2D(X):
        #  m, n = X.shape
        #  return np.array([ [ sum([ sum([ X[i,j]*np.exp(-1j*2*np.pi*(k_m*i/m + k_n*j/n)) for i in range(m) ]) for j in range(n) ]) for k_n in range(n) ] for k_m in range(m) ])
        '''
    def De_Padding(numpy_array_input):
        return numpy_array_input[0:ImageSize,0:ImageSize]

    img_input_np = np.array(img_input)
    padded_Filter = np.pad(filter_input, ((0, img_input_np.shape[0] - filter_input.shape[0]), (0, img_input_np.shape[1] - filter_input.shape[1])), 'constant')

    #Centering
    if centering:
        for a in range(0, img_input_np.shape[1]):
            for b in range(0, img_input_np.shape[0]):
                img_input_np[b, a] = img_input_np[b, a] * ((-1) ** (b + a))
                padded_Filter[b, a] = padded_Filter[b, a] * ((-1) ** (b + a))

    # fourier transformation
    padded_Filter_F_transformed_np = np.fft.fft2((padded_Filter))
    img_input_F_transformed_np = np.fft.fft2(img_input_np)

    output_arr = padded_Filter_F_transformed_np * img_input_F_transformed_np
    '''
    with np.printoptions(threshold=np.inf):
        print(output_arr)
        print(np.fft.ifft2(output_arr))
    '''
    output_arr = np.fft.ifft2(output_arr).real

    if De_padding:
        output_arr = De_Padding(output_arr)

    print(output_arr.shape)
    output = Image.fromarray(output_arr).convert('L')

    #Display
    if Display==True:
        # input image in spatial domain
        img_input.show()
        # input image in frequency domain
        Display_in_frequency_domain(img_input_np)
        # filter image in spatial domain
        Image.fromarray(padded_Filter * (filter_input.shape[0] * filter_input.shape[1] * 255)).show()
        # filter image in frequency domain
        Display_in_frequency_domain(padded_Filter)
        # final output image
        output.show()

    return output

##################################################
# main
##################################################

Active_Spatial=False
Active_Frequency=True

ImageIntegration_Spatial = [[A_1, A_2, A_3], ["S_No_Padding_", "S_Small_Padding", "S_Large_Padding"]]
ImageIntegration_Frequency = [[B_1, B_2, B_3], ["F_No_Padding_", "F_Small_Padding", "F_Large_Padding"]]
FilterIntegration = [[Filter_1, Filter_2, Filter_3, Filter_4],["41x41_avg", "7x7_avg", "41x41_gaussian", "7x7_gaussian"]]

##Image saving
Folder_Name = "Assignment_1_Output_Folder"

try:
    os.makedirs("./"+Folder_Name)
except FileExistsError:
    # directory already exists
    pass

#Original Data
A_1.save("./"+Folder_Name+"/" + "without_padding_original" + ".jpg")
A_2.save("./"+Folder_Name+"/" + "with_small_padding_original_for_Spatial_Filtering" + ".jpg")
A_3.save("./"+Folder_Name+"/" + "with_large_padding_original_for_Spatial_Filtering" + ".jpg")
B_2.save("./"+Folder_Name+"/" + "with_small_padding_original_for_Frequency_Filtering" + ".jpg")
B_3.save("./"+Folder_Name+"/" + "with_large_padding_original_for_Frequency_Filtering" + ".jpg")

#Spatial
if Active_Spatial:
    for a in range(len(ImageIntegration_Spatial[0])):
        for b in range(len(FilterIntegration[0])):
            output = Spatial_Convolution_Filtering(ImageIntegration_Spatial[0][a], FilterIntegration[0][b])
            output.save("./"+Folder_Name+"/" + ImageIntegration_Spatial[1][a] + FilterIntegration[1][b] + ".png")
#Frequency
if Active_Frequency:
    count = 0
    for a in range(len(ImageIntegration_Frequency[0])):
        for b in range(len(FilterIntegration[0])):
            count+=1
            output = Frequency_Domain_Multiplication_Filtering(ImageIntegration_Frequency[0][a], FilterIntegration[0][b])
            output.save("./"+Folder_Name+"/" + ImageIntegration_Frequency[1][a] + FilterIntegration[1][b] + ".jpg")
            print("Progess: "+ str(count) +"/" +str(len(ImageIntegration_Frequency[0])*len(FilterIntegration[0])))

print("--- %s seconds ---" % (time.time() - start_time))