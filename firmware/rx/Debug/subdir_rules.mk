################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Arm Compiler - building file: "$<"'
	"C:/ti/ti_cgt_arm_llvm_3.2.2.LTS/bin/tiarmclang.exe" -c -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mlittle-endian -mthumb -I"C:/Users/jim94/workspace_ccstheia/rfPacketRx" -I"C:/Users/jim94/workspace_ccstheia/rfPacketRx/Debug" -I"C:/ti/simplelink_cc13xx_cc26xx_sdk_8_32_00_07/source" -I"C:/ti/simplelink_cc13xx_cc26xx_sdk_8_32_00_07/kernel/tirtos7/packages" -I"C:/ti/simplelink_cc13xx_cc26xx_sdk_8_32_00_07/source/ti/posix/ticlang" -gdwarf-3 -march=armv7e-m -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/jim94/workspace_ccstheia/rfPacketRx/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

build-1502012906: ../rfPacketRx.syscfg
	@echo 'SysConfig - building file: "$<"'
	"C:/ti/sysconfig_1.21.1/sysconfig_cli.bat" -s "C:/ti/simplelink_cc13xx_cc26xx_sdk_8_32_00_07/.metadata/product.json" --script "C:/Users/jim94/workspace_ccstheia/rfPacketRx/rfPacketRx.syscfg" -o "syscfg" --compiler ticlang
	@echo 'Finished building: "$<"'
	@echo ' '

syscfg/ti_devices_config.c: build-1502012906 ../rfPacketRx.syscfg
syscfg/ti_radio_config.c: build-1502012906
syscfg/ti_radio_config.h: build-1502012906
syscfg/ti_drivers_config.c: build-1502012906
syscfg/ti_drivers_config.h: build-1502012906
syscfg/ti_utils_build_linker.cmd.genlibs: build-1502012906
syscfg/ti_utils_build_linker.cmd.genmap: build-1502012906
syscfg/ti_utils_build_compiler.opt: build-1502012906
syscfg/syscfg_c.rov.xs: build-1502012906
syscfg/ti_sysbios_config.h: build-1502012906
syscfg/ti_sysbios_config.c: build-1502012906
syscfg: build-1502012906

syscfg/%.o: ./syscfg/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Arm Compiler - building file: "$<"'
	"C:/ti/ti_cgt_arm_llvm_3.2.2.LTS/bin/tiarmclang.exe" -c -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mlittle-endian -mthumb -I"C:/Users/jim94/workspace_ccstheia/rfPacketRx" -I"C:/Users/jim94/workspace_ccstheia/rfPacketRx/Debug" -I"C:/ti/simplelink_cc13xx_cc26xx_sdk_8_32_00_07/source" -I"C:/ti/simplelink_cc13xx_cc26xx_sdk_8_32_00_07/kernel/tirtos7/packages" -I"C:/ti/simplelink_cc13xx_cc26xx_sdk_8_32_00_07/source/ti/posix/ticlang" -gdwarf-3 -march=armv7e-m -MMD -MP -MF"syscfg/$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/jim94/workspace_ccstheia/rfPacketRx/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


