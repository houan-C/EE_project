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
#include <string.h>
#include <ti/drivers/GPIO.h>
#include <ti/drivers/UART2.h> // 🐾 啟動聲帶：UART2 驅動
#include <ti/drivers/rf/RF.h>

#include DeviceFamily_constructPath(driverlib/rf_prop_mailbox.h)
#include "RFQueue.h"
#include "ti_drivers_config.h"
#include <ti_radio_config.h>


/***** Defines *****/
#define DATA_ENTRY_HEADER_SIZE 8
#define MAX_LENGTH 200       // 🐾 胃口對齊：與 TX 的 Payload 一致喵！
#define NUM_DATA_ENTRIES 4   // 增加緩衝區，影像傳輸才不會塞車喵！
#define NUM_APPENDED_BYTES 2 // 包含 1 Status Byte (RSSI)

/***** Variable declarations *****/
static RF_Object rfObject;
static RF_Handle rfHandle;
static UART2_Handle uart;

#pragma DATA_ALIGN(rxDataEntryBuffer, 4);
static uint8_t rxDataEntryBuffer[RF_QUEUE_DATA_ENTRY_BUFFER_SIZE(
    NUM_DATA_ENTRIES, MAX_LENGTH, NUM_APPENDED_BYTES)];

static dataQueue_t dataQueue;
static rfc_dataEntryGeneral_t *currentDataEntry;
static uint8_t packet[MAX_LENGTH + NUM_APPENDED_BYTES];

/***** Prototypes *****/
static void callback(RF_Handle h, RF_CmdHandle ch, RF_EventMask e);

/***** Function definitions *****/

void *mainThread(void *arg0) {
  /* 🐾 步驟一：UART2 高速通道初始化 */
  UART2_Params uartParams;
  UART2_Params_init(&uartParams);
  uartParams.baudRate = 921600; // 🐾 靈魂頻率：對齊 Python 的設定喵！
  uart = UART2_open(CONFIG_Display_0, &uartParams);

  GPIO_setConfig(CONFIG_GPIO_RLED, GPIO_CFG_OUT_STD | GPIO_CFG_OUT_LOW);
  GPIO_write(CONFIG_GPIO_RLED, CONFIG_GPIO_LED_OFF);

  /* 初始化 RF 接收隊列 */
  if (RFQueue_defineQueue(&dataQueue, rxDataEntryBuffer,
                          sizeof(rxDataEntryBuffer), NUM_DATA_ENTRIES,
                          MAX_LENGTH + NUM_APPENDED_BYTES)) {
    while (1)
      ; // 唔嗚～分配失敗了喵！
  }

  /* 🐾 步驟二：配置 RF 接收參數 (使用 SysConfig 生成的名字) */
  RF_cmdPropRx_custom868_0.pQueue = &dataQueue;
  RF_cmdPropRx_custom868_0.rxConf.bAutoFlushIgnored = 1;
  RF_cmdPropRx_custom868_0.rxConf.bAutoFlushCrcErr = 1;
  RF_cmdPropRx_custom868_0.maxPktLen = MAX_LENGTH;
  RF_cmdPropRx_custom868_0.pktConf.bRepeatOk = 1;
  RF_cmdPropRx_custom868_0.pktConf.bRepeatNok = 1;

  RF_Params rfParams;
  RF_Params_init(&rfParams);

  /* 開啟無線電並設定 923 MHz 頻率 */
  rfHandle =
      RF_open(&rfObject, &RF_prop_custom868_0,
              (RF_RadioSetup *)&RF_cmdPropRadioDivSetup_custom868_0, &rfParams);
  RF_postCmd(rfHandle, (RF_Op *)&RF_cmdFs_custom868_0, RF_PriorityNormal, NULL,
             0);

  /* 🐾 步驟三：進入永恆接收模式 */
  RF_runCmd(rfHandle, (RF_Op *)&RF_cmdPropRx_custom868_0, RF_PriorityNormal,
            &callback, RF_EventRxEntryDone);

  while (1)
    ;
}

/* 🐾 核心大腦：資料包裝 Callback */
void callback(RF_Handle h, RF_CmdHandle ch, RF_EventMask e) {
  if (e & RF_EventRxEntryDone) {
    GPIO_toggle(CONFIG_GPIO_RLED); // 收到碎片，眨眨眼喵！

    /* 抓取剛收到的一塊影像碎片 */
    currentDataEntry = RFQueue_getDataEntry();

    // 固定長度 200 Bytes，資料起點為 currentDataEntry->data
    uint8_t len = MAX_LENGTH;
    uint8_t *dataPtr = (uint8_t *)(&currentDataEntry->data);

    /* 🐾 關鍵打包術：包裝成 Python 認得的格式 */
    // [Len(1)] + [Data(200)] + [RSSI(1)] + [Dummy(1)]
    static uint8_t uartPacket[MAX_LENGTH + 3];
    int8_t rssi = (int8_t)dataPtr[len]; // 從 Payload 後面的 Status Byte 取得 RSSI

    uartPacket[0] = len;                  // 傳輸長度 (200)
    memcpy(&uartPacket[1], dataPtr, len); // 影像碎片數據
    uartPacket[len + 1] = (uint8_t)rssi;  // RSSI
    uartPacket[len + 2] = 0x00;           // 填充位元喵！

    /* 透過高速 UART 噴發給電腦！ */
    if (uart != NULL) {
      UART2_write(uart, uartPacket, len + 3, NULL);
    }

    /* 準備處理下一個影像碎片 */
    RFQueue_nextEntry();
  }
}