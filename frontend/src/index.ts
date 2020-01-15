// variables
let splash_view: HTMLElement = document.getElementById('splash_view');
let game_view: HTMLElement = document.getElementById('game_view');
let line_chart: HTMLElement = document.getElementById('line-chart-div');
let start_button: HTMLElement = document.getElementById('start_button');
let close_game_button: HTMLElement = document.getElementById('close_game_button');

// Add Event Listeners
start_button.addEventListener('click', () => showGameView())
close_game_button.addEventListener('click', () => hideGameView())

// Functions
function showGameView() {
    game_view.classList.add('visible');
    line_chart.classList.add('visible-chart');
    splash_view.classList.remove('visible');
}
function hideGameView() {
    game_view.classList.remove('visible');
    line_chart.classList.remove('visible-chart');
    splash_view.classList.add('visible');
}