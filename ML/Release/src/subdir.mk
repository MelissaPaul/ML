################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/LinearRegression.cpp \
../src/bigpoly.cpp \
../src/bigpolyarith.cpp \
../src/bigpolyarray.cpp \
../src/biguint.cpp \
../src/bt.cpp \
../src/chooser.cpp \
../src/ciphertext.cpp \
../src/decryptor.cpp \
../src/encoder.cpp \
../src/encryptionparams.cpp \
../src/encryptor.cpp \
../src/evaluationkeys.cpp \
../src/evaluator.cpp \
../src/keygenerator.cpp \
../src/plaintext.cpp \
../src/polycrt.cpp \
../src/randomgen.cpp \
../src/simulator.cpp \
../src/utilities.cpp 

OBJS += \
./src/LinearRegression.o \
./src/bigpoly.o \
./src/bigpolyarith.o \
./src/bigpolyarray.o \
./src/biguint.o \
./src/bt.o \
./src/chooser.o \
./src/ciphertext.o \
./src/decryptor.o \
./src/encoder.o \
./src/encryptionparams.o \
./src/encryptor.o \
./src/evaluationkeys.o \
./src/evaluator.o \
./src/keygenerator.o \
./src/plaintext.o \
./src/polycrt.o \
./src/randomgen.o \
./src/simulator.o \
./src/utilities.o 

CPP_DEPS += \
./src/LinearRegression.d \
./src/bigpoly.d \
./src/bigpolyarith.d \
./src/bigpolyarray.d \
./src/biguint.d \
./src/bt.d \
./src/chooser.d \
./src/ciphertext.d \
./src/decryptor.d \
./src/encoder.d \
./src/encryptionparams.d \
./src/encryptor.d \
./src/evaluationkeys.d \
./src/evaluator.d \
./src/keygenerator.d \
./src/plaintext.d \
./src/polycrt.d \
./src/randomgen.d \
./src/simulator.d \
./src/utilities.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


