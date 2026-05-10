/*
 *  ======== ti_radio_config.h ========
 *  Configured RadioConfig module definitions
 *
 *  DO NOT EDIT - This file is generated for the CC1352R1F3RGZ
 *  by the SysConfig tool.
 *
 *  Radio Config module version : 1.20.0
 *  SmartRF Studio data version : 2.32.0
 */
#ifndef _TI_RADIO_CONFIG_H_
#define _TI_RADIO_CONFIG_H_

#include <ti/devices/DeviceFamily.h>
#include DeviceFamily_constructPath(driverlib/rf_mailbox.h)
#include DeviceFamily_constructPath(driverlib/rf_common_cmd.h)
#include DeviceFamily_constructPath(driverlib/rf_prop_cmd.h)
#include <ti/drivers/rf/RF.h>

/* SmartRF Studio version that the RF data is fetched from */
#define SMARTRF_STUDIO_VERSION "2.32.0"

// *********************************************************************************
//   RF Frontend configuration
// *********************************************************************************
// RF design based on: LAUNCHXL-CC1352R1
#define LAUNCHXL_CC1352R1

// RF frontend configuration
#define FRONTEND_SUB1G_DIFF_RF
#define FRONTEND_SUB1G_EXT_BIAS
#define FRONTEND_24G_DIFF_RF
#define FRONTEND_24G_EXT_BIAS

// Supported frequency bands
#define SUPPORT_FREQBAND_868
#define SUPPORT_FREQBAND_2400

// TX power table size definitions
#define TXPOWERTABLE_868_PA13_SIZE 22 // 868 MHz, 13 dBm
#define TXPOWERTABLE_2400_PA5_SIZE 16 // 2400 MHz, 5 dBm

// TX power tables
extern RF_TxPowerTable_Entry txPowerTable_868_pa13[]; // 868 MHz, 13 dBm
extern RF_TxPowerTable_Entry txPowerTable_2400_pa5[]; // 2400 MHz, 5 dBm



//*********************************************************************************
//  RF Setting:   500 kbps, 190 kHz Deviation, 2-GFSK, 1150 kHz RX Bandwidth
//
//  PHY:          2gfsk500kbps154g
//  Setting file: setting_tc178_154g.json
//*********************************************************************************

// PA table usage
#define TX_POWER_TABLE_SIZE_2gfsk500kbps154g_0 TXPOWERTABLE_868_PA13_SIZE

#define txPowerTable_2gfsk500kbps154g_0 txPowerTable_868_pa13

// TI-RTOS RF Mode object
extern RF_Mode RF_prop_2gfsk500kbps154g_0;

// RF Core API commands
extern rfc_CMD_PROP_RADIO_DIV_SETUP_t RF_cmdPropRadioDivSetup_2gfsk500kbps154g_0;
extern rfc_CMD_FS_t RF_cmdFs_2gfsk500kbps154g_0;
extern rfc_CMD_PROP_TX_ADV_t RF_cmdPropTxAdv_2gfsk500kbps154g_0;
extern rfc_CMD_PROP_RX_ADV_t RF_cmdPropRxAdv_2gfsk500kbps154g_0;

// RF Core API overrides
extern uint32_t pOverrides_2gfsk500kbps154g_0[];

#endif // _TI_RADIO_CONFIG_H_
