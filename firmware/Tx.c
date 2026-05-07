#include <stdlib.h>
#include <unistd.h>

/* TI Drivers */
#include <ti/drivers/GPIO.h>
#include <ti/drivers/rf/RF.h>

// #include <ti/drivers/pin/PINCC26XX.h>

/* Driverlib Header files */
#include DeviceFamily_constructPath(driverlib / rf_prop_mailbox.h)

/* Board Header files */
#include "ti_drivers_config.h"
#include <ti_radio_config.h>

/***** Defines *****/
#include <ti/drivers/UART2.h>

/* Do power measurement */
// #define POWER_MEASUREMENT

/* Packet TX Configuration */
#define PAYLOAD_LENGTH 200
#ifdef POWER_MEASUREMENT
#define PACKET_INTERVAL                                                        \
  5 /* For power measurement set packet interval to 5s                         \
     */
#else
#define PACKET_INTERVAL 500000 /* Set packet interval to 500000us or 500ms */
#endif

/***** Prototypes *****/

/***** Variable declarations *****/
static RF_Object rfObject;
static RF_Handle rfHandle;

static uint8_t packet[PAYLOAD_LENGTH];
static uint16_t seqNumber;

/***** Function definitions *****/

void *mainThread(void *arg0) {
  RF_Params rfParams;
  RF_Params_init(&rfParams);

  /* UART2 初始化 */
  UART2_Handle uart;
  UART2_Params uartParams;
  UART2_Params_init(&uartParams);
  uartParams.baudRate = 921600; // 必須跟 Python 和 SysConfig 一樣喵！

  /* 打開你在 SysConfig 設定的那個 UART 接口 */
  uart = UART2_open(CONFIG_Display_0, &uartParams);

  GPIO_setConfig(CONFIG_GPIO_GLED, GPIO_CFG_OUT_STD | GPIO_CFG_OUT_LOW);
  GPIO_write(CONFIG_GPIO_GLED, CONFIG_GPIO_LED_OFF);

  /* RF 指令初始化 */
  RF_cmdPropTx_custom868_0.pktLen = PAYLOAD_LENGTH;
  RF_cmdPropTx_custom868_0.pPkt = packet;
  RF_cmdPropTx_custom868_0.startTrigger.triggerType = TRIG_NOW;

  /* 請求進入無線電模式 */
  rfHandle =
      RF_open(&rfObject, &RF_prop_custom868_0,
              (RF_RadioSetup *)&RF_cmdPropRadioDivSetup_custom868_0, &rfParams);

  /* 設定頻率 (也就是我們剛剛改的 923 MHz) */
  RF_postCmd(rfHandle, (RF_Op *)&RF_cmdFs_custom868_0, RF_PriorityNormal, NULL,
             0);

  while (1) {
    size_t bytesRead = 0;

    /* 🐾 步驟一：從 UART 讀取資料 */
    /* 這裡會卡住直到 Python 送來 812 Bytes 的 AVIF 碎片 */
    UART2_read(uart, packet, PAYLOAD_LENGTH, &bytesRead);

    if (bytesRead == PAYLOAD_LENGTH) {
      /* 🐾 步驟二：把讀到的資料透過 RF 噴發出去！ */
      RF_runCmd(rfHandle, (RF_Op *)&RF_cmdPropTx_custom868_0, RF_PriorityNormal,
                NULL, 0);

      /* 閃一下綠燈，代表成功轉發了一塊影像碎片喵！ */
      GPIO_toggle(CONFIG_GPIO_GLED);
    }

    /* 釋放一下射頻資源，稍微喘口氣 */
    RF_yield(rfHandle);
  }
}