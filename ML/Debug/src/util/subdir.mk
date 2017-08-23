################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/util/clipnormal.cpp \
../src/util/common.cpp \
../src/util/computation.cpp \
../src/util/crt.cpp \
../src/util/locks.cpp \
../src/util/mempool.cpp \
../src/util/modulus.cpp \
../src/util/ntt.cpp \
../src/util/polyarith.cpp \
../src/util/polyarithmod.cpp \
../src/util/polycore.cpp \
../src/util/polyextras.cpp \
../src/util/polyfftmult.cpp \
../src/util/polyfftmultmod.cpp \
../src/util/polymodulus.cpp \
../src/util/uintarith.cpp \
../src/util/uintarithmod.cpp \
../src/util/uintcore.cpp \
../src/util/uintextras.cpp 

OBJS += \
./src/util/clipnormal.o \
./src/util/common.o \
./src/util/computation.o \
./src/util/crt.o \
./src/util/locks.o \
./src/util/mempool.o \
./src/util/modulus.o \
./src/util/ntt.o \
./src/util/polyarith.o \
./src/util/polyarithmod.o \
./src/util/polycore.o \
./src/util/polyextras.o \
./src/util/polyfftmult.o \
./src/util/polyfftmultmod.o \
./src/util/polymodulus.o \
./src/util/uintarith.o \
./src/util/uintarithmod.o \
./src/util/uintcore.o \
./src/util/uintextras.o 

CPP_DEPS += \
./src/util/clipnormal.d \
./src/util/common.d \
./src/util/computation.d \
./src/util/crt.d \
./src/util/locks.d \
./src/util/mempool.d \
./src/util/modulus.d \
./src/util/ntt.d \
./src/util/polyarith.d \
./src/util/polyarithmod.d \
./src/util/polycore.d \
./src/util/polyextras.d \
./src/util/polyfftmult.d \
./src/util/polyfftmultmod.d \
./src/util/polymodulus.d \
./src/util/uintarith.d \
./src/util/uintarithmod.d \
./src/util/uintcore.d \
./src/util/uintextras.d 


# Each subdirectory must supply rules for building sources it contributes
src/util/%.o: ../src/util/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/libxml2 -I/usr/include/mlpack -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


