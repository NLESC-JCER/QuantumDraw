// variables
let splash_view: HTMLElement = document.getElementById('splash_view');
let ready_view: HTMLElement = document.getElementById('ready_view');
let game_view: HTMLElement = document.getElementById('game_view');
let banner: HTMLElement = document.getElementById('banner');

let line_chart: HTMLElement = document.getElementById('line-chart-div');
let start_button: HTMLElement = document.getElementById('start_button');

let ready_button: HTMLCollectionOf<Element> = ready_view.getElementsByClassName('button_with_shadow');
let close_game_button: HTMLElement = document.getElementById('close_game_button');

let tag_line: HTMLElement = document.getElementById('ready_tagline')

// Add Event Listeners
start_button.addEventListener('click', () => showReadyView())
Array.from(ready_button).forEach(function (element) {
    element.addEventListener('click', showGameView);
});
close_game_button.addEventListener('click', () => hideGameView())

// Functions
function showReadyView() {
    ready_view.classList.add('visible');
    splash_view.classList.remove('visible');
}

function showGameView() {
    banner.classList.add('shrink');
    game_view.classList.add('visible');
    line_chart.classList.add('visible-chart');
    ready_view.classList.remove('visible');
    splash_view.classList.remove('visible');
}
function hideGameView() {
    banner.classList.remove('shrink');
    game_view.classList.remove('visible');
    line_chart.classList.remove('visible-chart');
    ready_view.classList.add('visible');
}