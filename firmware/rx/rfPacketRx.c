/***** Includes *****/
#include <stdlib.h>
#include <string.h>
#include <ti/drivers/rf/RF.h>
#include <ti/drivers/GPIO.h>
#include <ti/drivers/UART2.h>
#include <ti/drivers/dpl/SemaphoreP.h> // 🐾 鈴鐺邏輯
#include DeviceFamily_constructPath(driverlib/rf_prop_mailbox.h)
#include "RFQueue.h"
#include "ti_drivers_config.h"
#include <ti_radio_config.h>

/***** Defines *****/
#define MAX_LENGTH             200 //
#define NUM_DATA_ENTRIES       16  // 🐾 加大隊列坑位，防塞車喵！
#define NUM_APPENDED_BYTES     2   

static RF_Object rfObject;
static RF_Handle rfHandle;
static UART2_Handle uart;
static SemaphoreP_Handle uartSem;

/* 🐾 資料緩存與隊列對齊 */
static uint8_t rxDataEntryBuffer[RF_QUEUE_DATA_ENTRY_BUFFER_SIZE(NUM_DATA_ENTRIES, MAX_LENGTH, NUM_APPENDED_BYTES)] __attribute__((aligned(4)));
static dataQueue_t dataQueue;
static rfc_dataEntryGeneral_t* currentDataEntry;

// 格式：[Len(1)] + [Data(200)] + [RSSI(1)] + [Dummy(1)] = 203
static uint8_t uartPacket[203]; 

static void callback(RF_Handle h, RF_CmdHandle ch, RF_EventMask e);

void *mainThread(void *arg0)
{
    /* 🐾 UART2 初始化 - 鮑率 921600 */
    UART2_Params uartParams;
    UART2_Params_init(&uartParams);
    uartParams.baudRate = 921600; 
    // 🐾 雖然 SysConfig 開啟了 Nonblocking，但我們在 Thread 裡依然可以使用 Blocking 讀寫邏輯
    uart = UART2_open(CONFIG_UART2_0, &uartParams);

    /* 🐾 初始化信號鈴鐺喵！ */
    uartSem = SemaphoreP_createBinary(0);

    GPIO_setConfig(CONFIG_GPIO_RLED, GPIO_CFG_OUT_STD | GPIO_CFG_OUT_LOW);

    if(RFQueue_defineQueue(&dataQueue, rxDataEntryBuffer, sizeof(rxDataEntryBuffer), 
                           NUM_DATA_ENTRIES, MAX_LENGTH + NUM_APPENDED_BYTES)) {
        while(1); 
    }

    /* 🐾 500kbps 進階配置 */
    RF_cmdPropRxAdv_2gfsk500kbps154g_0.pQueue = &dataQueue;
    RF_cmdPropRxAdv_2gfsk500kbps154g_0.maxPktLen = MAX_LENGTH;
    RF_cmdPropRxAdv_2gfsk500kbps154g_0.pktConf.bRepeatOk = 1;
    RF_cmdPropRxAdv_2gfsk500kbps154g_0.pktConf.bRepeatNok = 1;

    RF_Params rfParams;
    RF_Params_init(&rfParams);
    rfHandle = RF_open(&rfObject, &RF_prop_2gfsk500kbps154g_0, 
                       (RF_RadioSetup*)&RF_cmdPropRadioDivSetup_2gfsk500kbps154g_0, &rfParams);
    
    RF_postCmd(rfHandle, (RF_Op*)&RF_cmdFs_2gfsk500kbps154g_0, RF_PriorityNormal, NULL, 0);
    RF_postCmd(rfHandle, (RF_Op*)&RF_cmdPropRxAdv_2gfsk500kbps154g_0, RF_PriorityNormal, &callback, RF_EventRxEntryDone);

    while(1) {
        /* 🐾 等待 callback 搖鈴鐺 */
        SemaphoreP_pend(uartSem, SemaphoreP_WAIT_FOREVER);
        
        if (uart != NULL) {
            /* 從隊列抓取最新鮮的肉塊喵！ */
            currentDataEntry = RFQueue_getDataEntry();
            
            uint8_t len = MAX_LENGTH;
            uint8_t* dataPtr = (uint8_t*)(&currentDataEntry->data);
            int8_t rssi = (int8_t)dataPtr[len];

            // 🐾 打包協議封包
            uartPacket[0] = len;                     
            memcpy(&uartPacket[1], dataPtr, len);    
            uartPacket[len + 1] = (uint8_t)rssi;  
            uartPacket[len + 2] = 0x00;              

            /* 將 203 Bytes 噴進 4096 的 TX Ring Buffer 喵！ */
            UART2_write(uart, uartPacket, 203, NULL);

            /* 坑位釋放，準備接下一球 */
            RFQueue_nextEntry();
        }
    }
}

void callback(RF_Handle h, RF_CmdHandle ch, RF_EventMask e)
{
    if (e & RF_EventRxEntryDone)
    {
        GPIO_toggle(CONFIG_GPIO_RLED); 
        /* 🐾 搖鈴鐺通知後台處理，callback 絕對不准停下來喵！ */
        SemaphoreP_post(uartSem);
    }
}