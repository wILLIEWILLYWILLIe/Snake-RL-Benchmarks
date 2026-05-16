// Snake Game "driver"
// Runs the snake game on top of the LED matrix

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "nrfx_gpiote.h"

#include "led_matrix.h"
#include "microbit_v2.h"

// represent snake body part locations
typedef struct {
  int8_t row;
  int8_t col;
} snakeloc_t;

// represent movement direction of the snake
typedef enum {
  UP = 1,
  LEFT,
  DOWN,
  RIGHT,
} snakedir_t;

// collecting the many variables we need to track the game's state
//  In C++ we'd just make a class, but alas we're in C...
typedef struct {
  // model of the current game state
  //  True means a snake body piece is in that location
  //  False means nothing is in that location
  bool model[5][5];

  // current position of the head of the snake
  snakeloc_t head;

  // current direction the snake is traveling in
  snakedir_t dir;

  // button pressed since last iteration
  //  0 means no button has been pressed
  //  Otherwise set to BTN_A or BTN_B
  uint8_t button_pressed;

  // iterations since the game has started (for high score)
  // also used to determine when to grow the snake
  uint8_t iterations;

  // whether the game is still running or not
  bool game_over;
} game_state_t;

static game_state_t game_state;

// ***** Queue implementation *****

// TODO
// You will need init(), push(), and pop() functionality.
// This is most reasonably implemented as a ring buffer using a 25-element
// array, although other implementations are possible.

static snakeloc_t snake_queue[25];
static int head_ptr = 0;
static int tail_ptr = 0;
static int cur_size = 0;

static bool is_full(void) { return cur_size == 25; }

static void enqueue(snakeloc_t loc) {
  if (is_full()) {
    return;
  }
  snake_queue[head_ptr] = loc;
  head_ptr = (head_ptr + 1) % 25;
  cur_size++;
}

static snakeloc_t dequeue(void) {
  snakeloc_t tail = snake_queue[tail_ptr];
  tail_ptr = (tail_ptr + 1) % 25;
  cur_size--;
  return tail;
}
// ********************************

// button press callback function
// you should not need to edit this, but you can if you want to
static void gpio_handler(nrfx_gpiote_pin_t pin, nrf_gpiote_polarity_t _unused) {
  // overwrite the current button pressed as this one
  // if multiple buttons have been pressed, only the most recent is saved
  game_state.button_pressed = (uint8_t)pin;
}

// intialize the game state
// you should not need to edit this, but you can if you want to
void snake_game_init(void) {
  // initialize game state
  memset(game_state.model, false, 5 * 5 * sizeof(bool));
  game_state.head.row = 2;
  game_state.head.col = 0;
  game_state.dir = RIGHT;
  game_state.button_pressed = 0;
  game_state.iterations = 0;
  game_state.game_over = false;

  // initial snake position
  game_state.model[game_state.head.row][game_state.head.col] = true;

  // initialize ring buffer
  // push initial head into the queue
  // TODO
  head_ptr = 0;
  tail_ptr = 0;
  cur_size = 0;
  enqueue(game_state.head);

  // initialize button inputs
  // calls the `gpio_handler` callback function when either button is pressed
  nrfx_gpiote_init();
  nrfx_gpiote_in_config_t in_config =
      NRFX_GPIOTE_CONFIG_IN_SENSE_LOTOHI(true); // high-accuracy mode
  nrfx_gpiote_in_init(BTN_A, &in_config, gpio_handler);
  nrfx_gpiote_in_event_enable(BTN_A, true); // enable interrupts for BTN_A
  nrfx_gpiote_in_init(BTN_B, &in_config, gpio_handler);
  nrfx_gpiote_in_event_enable(BTN_B, true); // enable interrupts for BTN_B
}

static void snake_gameover_flash(void) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      game_state.model[i][j] = true;
    }
  }

  update_led_states(game_state.model);
  char score_str[16];
  sprintf(score_str, "%d", game_state.iterations);
  led_matrix_display_string(score_str, true, 0);
}

// advance the game state
// this is intended to be called repeatedly from the main code, with a delay
// between each call each call advances the game state one iteration
void snake_game_advance_state(void) {
  if (game_state.game_over) {
    return;
  }

  // 1. Handle button presses
  if (game_state.button_pressed == BTN_A) {
    // Left
    game_state.dir = (game_state.dir % 4) + 1;
  } else if (game_state.button_pressed == BTN_B) {
    // Right
    game_state.dir = ((game_state.dir + 2) % 4) + 1;
  }
  game_state.button_pressed = 0;

  // 2. Calculate new head location
  snakeloc_t next_head = game_state.head;
  if (game_state.dir == UP) {
    next_head.row -= 1;
  } else if (game_state.dir == DOWN) {
    next_head.row += 1;
  } else if (game_state.dir == LEFT) {
    next_head.col -= 1;
  } else if (game_state.dir == RIGHT) {
    next_head.col += 1;
  }

  // 3. Check for collisions
  if (next_head.row < 0 || next_head.row > 4 || next_head.col < 0 ||
      next_head.col > 4 || game_state.model[next_head.row][next_head.col]) {
    game_state.game_over = true;
    snake_gameover_flash();
    printf("Game Over! Final Score (Iterations): %d\n", game_state.iterations);
    return;
  }

  // 4. Update head position and iterations
  game_state.head = next_head;
  game_state.iterations++;

  // 5. Update model and queue
  game_state.model[game_state.head.row][game_state.head.col] = true;
  enqueue(game_state.head);

  // Growth logic: grow every 10 iterations
  if (game_state.iterations % 10 != 0) {
    snakeloc_t tail = dequeue();
    game_state.model[tail.row][tail.col] = false;
  }

  // 6. Update display
  update_led_states(game_state.model);
}