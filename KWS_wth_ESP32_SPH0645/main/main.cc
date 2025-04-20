/*
 * SPDX-FileCopyrightText: 2010-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

extern "C" {

#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>

#include "nvs_flash.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "esp_app_trace.h" /* Used with System view SEGGER */

#include "soc/soc.h" //--> disable brownout problems
#include "soc/rtc_cntl_reg.h"  //--> disable brownout problems
#include "KWS/keyword_spotting_interface.h"
#include "driver/gpio.h"
}



extern "C" void app_main(void)
{

    printf("Hello world!\n");

    // /* Init pins */
    // gpio_set_direction( GO_LED_PIN    , GPIO_MODE_OUTPUT );
    // gpio_set_direction( STOP_LED_PIN  , GPIO_MODE_OUTPUT );
    // gpio_set_direction( LEFT_LED_PIN  , GPIO_MODE_OUTPUT );
    // gpio_set_direction( RIGHT_LED_PIN , GPIO_MODE_OUTPUT );

    // gpio_set_level( GO_LED_PIN    , 0 ); /* Turn LED on */
    // gpio_set_level( STOP_LED_PIN  , 0 ); /* Turn LED on */
    // gpio_set_level( LEFT_LED_PIN  , 0 ); /* Turn LED on */
    // gpio_set_level( RIGHT_LED_PIN , 0 ); /* Turn LED on */        

    /* Start keyword spotting task */
    keyword_spotting_app_start();
   
}
