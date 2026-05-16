// LED Matrix app
//
// Display messages on the LED matrix

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "nrf_delay.h"

#include "led_matrix.h"
#include "microbit_v2.h"
#include "snake_game.h"

int main(void) {
  printf("Board started!\n");

  // initialize LED matrix driver
  led_matrix_init();

  // call other functions here

  /////////// Single char
  // led_matrix_display_char('2');

  /////////// Arbitrary string
  // led_matrix_display_string("Hi CE346!", false, 0);
  // nrf_delay_ms(2000);
  // led_matrix_display_string("It works!", true, 1);

  /////////// test states for update function
  // bool test_states[5][5] = {false};
  // test_states[0][1] = true;
  // test_states[0][3] = true;
  // test_states[1][0] = true;
  // test_states[1][2] = true;
  // test_states[1][4] = true;
  // test_states[2][0] = true;
  // test_states[2][4] = true;
  // test_states[3][1] = true;
  // test_states[3][3] = true;
  // test_states[4][2] = true;
  // update_led_states(test_states);

  /////////// Snake game
  snake_game_init();

  // loop forever
  while (1) {
    snake_game_advance_state();
    nrf_delay_ms(500);
  }
}