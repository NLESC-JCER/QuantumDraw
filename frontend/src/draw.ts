import * as d3 from 'd3';
import * as chartXkcd from 'chart.xkcd';
import XY from './XY';
import './index.css';

declare const window: any;

const root = `${window.location.host}${window.location.pathname}`;
const websocketURL = `ws://${root}ws`;
const socket = window.socket = new WebSocket('ws://localhost:8888/ws');
// const socket = window.socket = new WebSocket(websocketURL);

let canvas: HTMLCanvasElement;
let context: CanvasRenderingContext2D;

// let level = 0;
let userGuessAttemptNumber = 0;

let userscore = 0;
let aiscore = 0;

let startTime = 0;

const HEIGHT = parseInt(d3.select('#canvas_view').style('width'), 10);
const WIDTH = parseInt(d3.select('#canvas_view').style('width'), 10) - 40;
const LINEWIDTH = 1;
const STROKESTYLE = "#000000";
const TARGETSTYLE = "#ff0000";
const AISTYLE = "#0000ff";

canvas = document.createElement('canvas');
canvas.width = WIDTH;
canvas.height = HEIGHT;
document.getElementById('canvas_view').appendChild(canvas);
context = canvas.getContext("2d");

const strokes = [];
let potential = [];
let aiguess = [];

let linechartData = {
    datasets: [{
        label: 'User scores',
        data: [],
    }, {
        label: 'AI scores',
        data: [],
    }],
};

(window as any).strokes = strokes;
const curve = d3.curveBasis(context);
const redo = [];

function render() {
    context.clearRect(0, 0, WIDTH, HEIGHT);
    for (const stroke of strokes) {
        context.beginPath();
        curve.lineStart();
        for (const point of stroke) {
            curve.point(point[0], point[1]);
        }
        if (stroke.length === 1) curve.point(stroke[0][0], stroke[0][1]);
        curve.lineEnd();
        context.lineWidth = LINEWIDTH;
        context.strokeStyle = STROKESTYLE;
        context.stroke();
    }

    for (const potentialStroke of potential) {
        context.beginPath();
        curve.lineStart();
        for (const point of potentialStroke) {
            curve.point(x(point[0]), y(point[1]));
        }
        if (potentialStroke.length === 1) curve.point(potentialStroke[0][0], potentialStroke[0][1]);
        curve.lineEnd();
        context.lineWidth = LINEWIDTH;
        context.strokeStyle = TARGETSTYLE;
        context.stroke();
    }

    for (const aiStroke of aiguess) {
        context.beginPath();
        curve.lineStart();
        for (const point of aiStroke) {
            curve.point(x(point[0]), y(point[1]));
        }
        if (aiStroke.length === 1) curve.point(aiStroke[0][0], aiStroke[0][1]);
        curve.lineEnd();
        context.lineWidth = LINEWIDTH;
        context.strokeStyle = AISTYLE;
        context.stroke();
    }

    // context.canvas.value = strokes;
    context.canvas.dispatchEvent(new CustomEvent("input"));
}

var x = d3.scaleLinear()
    .domain([-5, 5])
    .range([-1, WIDTH + 1]);

var y = d3.scaleLinear()
    .domain([1, 0])
    .range([-1, HEIGHT + 1]);


d3.select(context.canvas).call(d3.drag()
    .container(context.canvas)
    .subject(dragsubject)
    .on("end", () => {
        sendGuess(strokes[0].map(stroke => [x.invert(stroke[0]), y.invert(stroke[1])]));
    })
    .on("start drag", dragged)
    .on("start.render drag.render", render))

// Create a new empty stroke at the start of a drag gesture.
function dragsubject() {
    strokes.length = 0;
    const stroke = [];

    strokes.push(stroke);

    return stroke;
}

// Add to the stroke when dragging.
function dragged() {
    let newX = d3.event.x;
    const stroke = d3.event.subject;
    if (stroke) {
        let maxX = Math.max(...stroke.map(strokeSegment => strokeSegment[0]));
        if (newX > maxX) {
            stroke.push([d3.event.x, d3.event.y]);
        }
    }
}

function reset() {
    strokes.length = 0;
    sendResetMessage(0);
    userGuessAttemptNumber = 0;

    linechartData = {
        datasets: [{
            label: 'User scores',
            data: [],
        }, {
            label: 'AI scores',
            data: [],
        }],
    };
    window.linechart.data = linechartData;

    // Clear and Rerender
    document.querySelector('.line-chart>g:first-child').innerHTML = '';
    window.linechart.render();

    render();
}

window.addEventListener('load', () => {
    reset();
});

function updateUserScore(time: number, value: number) {
    const userScoreP = d3.select('.userScoreP');
    const userAttemptP = d3.select('.userAttemptP');
    const userTimeP = d3.select('.userTimeP');

    if (userGuessAttemptNumber === 0) {
        startTime = time;
    }

    let displayTime = (time - startTime) / 1000;
    userGuessAttemptNumber++;

    userscore = value * 1000;

    userScoreP.text("User score now: " + userscore.toString());
    userAttemptP.text("User attempt: " + userGuessAttemptNumber.toString());
    userTimeP.text("User time: " + displayTime.toString());

    linechartData.datasets[0].data.push({ x: displayTime, y: userscore });

    // Clear and Rerender
    document.querySelector('.line-chart>g:first-child').innerHTML = '';
    window.linechart.render();


}

function updateAIScore(time: number, value: number) {
    const aiscoreP = d3.select('.aiScoreP');

    let displayTime = (time - startTime) / 1000;

    aiscore = value * 1000;

    aiscoreP.text("AI score now: " + aiscore.toString());

    linechartData.datasets[1].data.push({ x: displayTime, y: aiscore });
    document.querySelector('.line-chart>g:first-child').innerHTML = '';
    window.linechart.render();

}

// createScore();

//Add reset button
const button = document.createElement('button');
button.innerHTML = 'reset';
button.addEventListener('click', reset as any);
//document.body.appendChild(button);

socket.addEventListener('message', function (event) {
    const time = new Date().getTime();
    const parsedData = JSON.parse(event.data);
    const eventType = parsedData.type;

    if (eventType === 'potential') {
        potential = [parsedData.data];
        render();
    } else if (eventType === 'ai_score') {
        if (userGuessAttemptNumber != 0) {
            updateAIScore(time, parsedData.score);
            aiguess = [parsedData.points];
            render();
        }
    } else if (eventType === 'user_score') {
        updateUserScore(time, parsedData.score);
    }
})
//
// socket.addEventListener("gameover", function(event: any) {
//     const time = JSON.parse(event.data).time;
//     aiscore = JSON.parse(event.data).aiscore;
//     userscore = JSON.parse(event.data).userscore;
// });

interface Message {
    type: string;
    data: any;
}

function sendResetMessage(level: number = 0) {
    let message: Message = {
        type: 'reset',
        data: level
    };
    socket.send(JSON.stringify(message));
}

function sendGuess(data: Array<Array<number>>) {
    let message: Message = {
        type: 'guess',
        data: data
    };
    socket.send(JSON.stringify(message));
}

window.addEventListener('load', () => {
    const svg = document.querySelector('.line-chart');

    window.linechart = new chartXkcd.XY(
        svg, {
        title: 'Your score VS AI score', // optional
        xLabel: 'Time in seconds', // optional
        yLabel: 'Score', // optional
        data: linechartData,
        options: { // optional
            xTickCount: 10,
            yTickCount: 10,
            dotSize: 1,
            showLine: true,
            legendPosition: chartXkcd.config.positionType.upLeft
        }
    });
    canvas.height = parseInt(d3.select('.line-chart').style('height'), 10);
})
window.addEventListener("orientationchange", function () {
    window.location.reload();
});

