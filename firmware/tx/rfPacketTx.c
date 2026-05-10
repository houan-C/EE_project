/*
 * Copyright (c) 2019, Texas Instruments Incorporated
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * *  Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * *  Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/***** Includes *****/
#include <stdlib.h>
#include <unistd.h>
#include <ti/drivers/rf/RF.h>
#include <ti/drivers/GPIO.h>
#include <ti/drivers/UART2.h> 
#include DeviceFamily_constructPath(driverlib/rf_prop_mailbox.h)
#include "ti_drivers_config.h"
#include <ti_radio_config.h>

/***** Defines *****/
#define PAYLOAD_LENGTH      200 // 🐾 胃口對齊

static RF_Object rfObject;
static RF_Handle rfHandle;
static uint8_t packet[PAYLOAD_LENGTH];

void *mainThread(void *arg0)
{
    /* 🐾 UART2 初始化 - 請在 SysConfig 命名為 CONFIG_UART2_0 */
    UART2_Handle uart;
    UART2_Params uartParams;
    UART2_Params_init(&uartParams);
    uartParams.baudRate = 921600; // 靈魂頻率
    uart = UART2_open(CONFIG_UART2_0, &uartParams);

    GPIO_setConfig(CONFIG_GPIO_GLED, GPIO_CFG_OUT_STD | GPIO_CFG_OUT_LOW);

    /* RF 指令配置 - 固定長度模式 */
    RF_cmdPropTxAdv_2gfsk500kbps154g_0.pktLen = PAYLOAD_LENGTH;
    RF_cmdPropTxAdv_2gfsk500kbps154g_0.pPkt = packet;
    RF_cmdPropTxAdv_2gfsk500kbps154g_0.startTrigger.triggerType = TRIG_NOW;

    RF_Params rfParams;
    RF_Params_init(&rfParams);
    rfHandle = RF_open(&rfObject, &RF_prop_2gfsk500kbps154g_0, 
                       (RF_RadioSetup*)&RF_cmdPropRadioDivSetup_2gfsk500kbps154g_0, &rfParams);
    
    /* 注入 923 MHz 的能量喵！ */
    RF_postCmd(rfHandle, (RF_Op*)&RF_cmdFs_2gfsk500kbps154g_0, RF_PriorityNormal, NULL, 0);

    while(1)
    {
        size_t bytesRead = 0;
        /* 🐾 阻塞讀取，直到 Python 給滿 200 Bytes */
        UART2_read(uart, packet, PAYLOAD_LENGTH, &bytesRead);

        if (bytesRead == PAYLOAD_LENGTH)
        {
            RF_runCmd(rfHandle, (RF_Op*)&RF_cmdPropTxAdv_2gfsk500kbps154g_0, RF_PriorityNormal, NULL, 0);
            GPIO_toggle(CONFIG_GPIO_GLED); // 成功噴發碎片，閃綠燈喵！
        }
    }
}