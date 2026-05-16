// LED Matrix Driver
// Displays characters on the LED matrix

#include <stdbool.h>
#include <stdio.h>

#include "app_timer.h"
#include "nrf_gpio.h"

#include "font.h"
#include "led_matrix.h"
#include "microbit_v2.h"

APP_TIMER_DEF(my_timer1);
APP_TIMER_DEF(char_timer); // second timer to control the next char

static const char *display_string = NULL;
static uint16_t display_index = 0;
static volatile bool display_done = false;

static uint32_t col_flash_freq = 32768 / 500;
static uint32_t char_display_time = 32768 / 2;

static bool use_marquee = false;
static uint16_t scroll_offset = 0;
static bool any_char_visible = false;
static int repeat_times = 0;

static uint32_t row_pins[] = {LED_ROW1, LED_ROW2, LED_ROW3, LED_ROW4, LED_ROW5};
static uint32_t col_pins[] = {LED_COL1, LED_COL2, LED_COL3, LED_COL4, LED_COL5};

// 2D matrix of boolean values
static bool led_states[5][5] = {false};
static uint8_t current_row = 0;

void update_led_states(bool new_states[5][5]) {
  memcpy(led_states, new_states, sizeof(led_states));
}

void led_matrix_display_char(char c) {
  // printf("size of font: %d", sizeof(font));
  if (c < 0 || c > 127)
    return;
  for (uint8_t i = 0; i < 5; i++) {
    // printf("%x - %x %x \n", (int)c, (int)font[(int)c][i], font[(int)c][i]);
    uint8_t row_data = font[(int)c][i];
    for (uint8_t j = 0; j < 5; j++) {
      led_states[i][j] = (row_data >> j) & 1;
    }
  }
}

void led_matrix_display_char_marquee() {
  any_char_visible = false;
  for (int col = 0; col < 5; col++) {
    int total_col = scroll_offset + col;
    int char_idx = total_col / 5;
    int char_col = total_col % 5;

    if (char_idx < 0 ||
        char_idx >= strlen(display_string)) { // deal with empty idx
      for (int row = 0; row < 5; row++) {
        led_states[row][col] = false;
      }
    } else {
      any_char_visible = true;
      if (char_col == 5) { // deal with space between characters
        for (int row = 0; row < 5; row++) {
          led_states[row][col] = false;
        }
      } else {
        for (int row = 0; row < 5; row++) {
          uint8_t row_data = font[(int)display_string[char_idx]][row];
          led_states[row][col] = (row_data >> char_col) & 1;
        }
      }
    }
  }
}

static void handle_row(uint8_t row) {
  for (uint8_t i = 0; i < 5; i++) {
    nrf_gpio_pin_clear(row_pins[i]);
  }
  nrf_gpio_pin_write(row_pins[row], 1);
  for (uint8_t i = 0; i < 5; i++) {
    nrf_gpio_pin_write(col_pins[i], !led_states[row][i]);
  }
}

static void swap_state(void *_unused) {
  handle_row(current_row);
  current_row++;
  if (current_row == 5) {
    current_row = 0;
  }
}

static void char_timer_handler(void *_unused) {
  if (use_marquee) {
    led_matrix_display_char_marquee();
    if (!any_char_visible && scroll_offset > 0) {
      if (repeat_times) {
        scroll_offset = 0;
        repeat_times--;
      } else {
        app_timer_stop(char_timer);
        display_done = true;
      }
      return;
    }
    scroll_offset++;

  } else {
    if (display_string == NULL || display_string[display_index] == '\0') {
      if (repeat_times) {
        display_index = 0;
        repeat_times--;
      } else {
        app_timer_stop(char_timer);
        display_done = true;
      }
      return;
    }
    led_matrix_display_char(display_string[display_index]);
    display_index++;
  }
}

static void create_X_pattern() {
  led_states[0][0] = true;
  led_states[0][4] = true;
  led_states[1][1] = true;
  led_states[1][3] = true;
  led_states[2][2] = true;
  led_states[3][1] = true;
  led_states[3][3] = true;
  led_states[4][0] = true;
  led_states[4][4] = true;
}

void led_matrix_display_string(const char *str, bool marquee, int num_repeats) {
  display_string = str;
  display_index = 0;
  display_done = false;
  use_marquee = marquee;
  scroll_offset = 0;
  repeat_times = num_repeats;

  if (use_marquee) {
    app_timer_start(char_timer, char_display_time / 2, NULL);
  } else {
    if (display_string[display_index] != '\0') {
      led_matrix_display_char(display_string[display_index]);
      display_index++;
      app_timer_start(char_timer, char_display_time, NULL);
    } else {
      display_done = true;
      return;
    }
  }

  while (!display_done) { // blocking interface
    // __WFE();
  }
}

void led_matrix_init(void) {
  // initialize row pins
  for (uint8_t i = 0; i < 5; i++) {
    nrf_gpio_pin_dir_set(row_pins[i], NRF_GPIO_PIN_DIR_OUTPUT);
    nrf_gpio_pin_clear(row_pins[i]);
  }

  // initialize col pins
  for (uint8_t i = 0; i < 5; i++) {
    nrf_gpio_pin_dir_set(col_pins[i], NRF_GPIO_PIN_DIR_OUTPUT);
    nrf_gpio_pin_set(col_pins[i]);
  }

  // set default values for pins
  nrf_gpio_pin_set(LED_ROW1);
  nrf_gpio_pin_clear(LED_COL1);
  nrf_gpio_pin_clear(LED_COL2);
  nrf_gpio_pin_clear(LED_COL3);
  nrf_gpio_pin_clear(LED_COL4);
  nrf_gpio_pin_set(LED_COL5);

  // initialize timer(s) (Step 2 and onwards)
  app_timer_init();
  // create_X_pattern();
  app_timer_create(&my_timer1, APP_TIMER_MODE_REPEATED, swap_state);
  app_timer_start(my_timer1, col_flash_freq, NULL);
  app_timer_create(&char_timer, APP_TIMER_MODE_REPEATED, char_timer_handler);

  // set default state for the LED display (Step 3 and onwards)
}