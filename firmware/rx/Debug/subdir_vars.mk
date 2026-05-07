################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Add inputs and outputs from these tool invocations to the build variables 
CMD_SRCS += \
../cc13x2_cc26x2_tirtos7.cmd 

SYSCFG_SRCS += \
../rfPacketRx.syscfg 

C_SRCS += \
../RFQueue.c \
../main_tirtos.c \
../rfPacketRx.c \
./syscfg/ti_devices_config.c \
./syscfg/ti_radio_config.c \
./syscfg/ti_drivers_config.c \
./syscfg/ti_sysbios_config.c 

GEN_FILES += \
./syscfg/ti_devices_config.c \
./syscfg/ti_radio_config.c \
./syscfg/ti_drivers_config.c \
./syscfg/ti_utils_build_compiler.opt \
./syscfg/ti_sysbios_config.c 

GEN_MISC_DIRS += \
./syscfg 

C_DEPS += \
./RFQueue.d \
./main_tirtos.d \
./rfPacketRx.d \
./syscfg/ti_devices_config.d \
./syscfg/ti_radio_config.d \
./syscfg/ti_drivers_config.d \
./syscfg/ti_sysbios_config.d 

GEN_OPTS += \
./syscfg/ti_utils_build_compiler.opt 

OBJS += \
./RFQueue.o \
./main_tirtos.o \
./rfPacketRx.o \
./syscfg/ti_devices_config.o \
./syscfg/ti_radio_config.o \
./syscfg/ti_drivers_config.o \
./syscfg/ti_sysbios_config.o 

GEN_MISC_FILES += \
./syscfg/ti_radio_config.h \
./syscfg/ti_drivers_config.h \
./syscfg/ti_utils_build_linker.cmd.genlibs \
./syscfg/ti_utils_build_linker.cmd.genmap \
./syscfg/syscfg_c.rov.xs \
./syscfg/ti_sysbios_config.h 

GEN_MISC_DIRS__QUOTED += \
"syscfg" 

OBJS__QUOTED += \
"RFQueue.o" \
"main_tirtos.o" \
"rfPacketRx.o" \
"syscfg\ti_devices_config.o" \
"syscfg\ti_radio_config.o" \
"syscfg\ti_drivers_config.o" \
"syscfg\ti_sysbios_config.o" 

GEN_MISC_FILES__QUOTED += \
"syscfg\ti_radio_config.h" \
"syscfg\ti_drivers_config.h" \
"syscfg\ti_utils_build_linker.cmd.genlibs" \
"syscfg\ti_utils_build_linker.cmd.genmap" \
"syscfg\syscfg_c.rov.xs" \
"syscfg\ti_sysbios_config.h" 

C_DEPS__QUOTED += \
"RFQueue.d" \
"main_tirtos.d" \
"rfPacketRx.d" \
"syscfg\ti_devices_config.d" \
"syscfg\ti_radio_config.d" \
"syscfg\ti_drivers_config.d" \
"syscfg\ti_sysbios_config.d" 

GEN_FILES__QUOTED += \
"syscfg\ti_devices_config.c" \
"syscfg\ti_radio_config.c" \
"syscfg\ti_drivers_config.c" \
"syscfg\ti_utils_build_compiler.opt" \
"syscfg\ti_sysbios_config.c" 

C_SRCS__QUOTED += \
"../RFQueue.c" \
"../main_tirtos.c" \
"../rfPacketRx.c" \
"./syscfg/ti_devices_config.c" \
"./syscfg/ti_radio_config.c" \
"./syscfg/ti_drivers_config.c" \
"./syscfg/ti_sysbios_config.c" 

SYSCFG_SRCS__QUOTED += \
"../rfPacketRx.syscfg" 


