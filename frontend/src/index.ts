// variables
let splash_view: HTMLElement = document.getElementById('splash_view');
let ready_view: HTMLElement = document.getElementById('ready_view');
let game_view: HTMLElement = document.getElementById('game_view');


let line_chart: HTMLElement = document.getElementById('line-chart-div');
let start_button: HTMLElement = document.getElementById('start_button');
let ready_button: HTMLElement = document.getElementById('ready_button');
let next_button: HTMLElement = document.getElementById('next_button');
let close_game_button: HTMLElement = document.getElementById('close_game_button');

let tag_line: HTMLElement = document.getElementById('ready_tagline')

// Add Event Listeners
start_button.addEventListener('click', () => showReadyView())
next_button.addEventListener('click', () => showNextView())
ready_button.addEventListener('click', () => showGameView())
close_game_button.addEventListener('click', () => hideGameView())

// Functions
function showReadyView() {
    ready_view.classList.add('visible');
    splash_view.classList.remove('visible');
}

function showNextView() {
    hideGameView()
    showReadyView()
    tag_line.innerHTML = '<span class="tagline_font"> Solve the Morse oscillator in less than 60 sec.</span> <button id="ready_button" class="button_with_shadow">Ready ?</button>'
    console.log(tag_line)
}

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