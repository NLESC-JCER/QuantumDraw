import * as d3 from 'd3';

declare const window: any;

const root = `${window.location.host}${window.location.pathname}`;
const websocketURL = `ws://${root}ws`;
// const socket = window.socket = new WebSocket('ws://localhost:8888/ws');
const socket = window.socket = new WebSocket(websocketURL);

let canvas: HTMLCanvasElement;
let context: CanvasRenderingContext2D;

// let level = 0;
let guessAttemptNumber = 0;

let userscore = 0;
let aiscore = 0;

const HEIGHT = 500;
const WIDTH = 1000;
const LINEWIDTH = 1;
const STROKESTYLE = "#000000";
const TARGETSTYLE = "#ff0000";
const AISTYLE = "#0000ff";

canvas = document.createElement('canvas');
canvas.width = WIDTH;
canvas.height = HEIGHT;

document.body.appendChild(canvas);
context = canvas.getContext("2d");

const strokes = [[[79, 208], [105, 204], [151, 189], [264, 159], [400, 119], [509, 95], [567, 91], [582, 95], [585, 111], [585, 136], [585, 164], [586, 184], [595, 198], [608, 201], [647, 201], [723, 188], [811, 169], [881, 154], [906, 146], [915, 142], [916, 144], [916, 147]]];
let potential = [[[79, 208], [105, 204], [151, 189], [264, 159], [400, 119], [509, 95], [567, 91], [582, 95], [585, 111], [585, 136], [585, 164], [586, 184], [595, 198], [608, 201], [647, 201], [723, 188], [811, 169], [881, 154], [906, 146], [915, 142], [916, 144], [916, 147]]];
let aiguess = [[[33, 271], [32, 264], [32, 245], [41, 212], [56, 165], [73, 103], [85, 69], [92, 51], [97, 42], [99, 38], [101, 37], [104, 37], [108, 40], [111, 54], [126, 81], [141, 112], [154, 139], [164, 165], [173, 185], [179, 200], [185, 214], [187, 219], [190, 222], [191, 223], [193, 222], [203, 205], [230, 168], [266, 136], [296, 117], [315, 107], [324, 103], [326, 103], [328, 108], [330, 123], [341, 161], [354, 208], [361, 230], [368, 244], [371, 251], [373, 253], [375, 253], [388, 237], [414, 202], [444, 160], [495, 109], [518, 92], [527, 85], [529, 84], [532, 85], [536, 106], [543, 155], [551, 210], [560, 255], [570, 281], [580, 298], [593, 306], [620, 303], [633, 275], [655, 226], [683, 174], [709, 141], [726, 125], [736, 118], [738, 117], [741, 117], [745, 130], [750, 158], [754, 182], [758, 200], [762, 209], [768, 209], [787, 194], [819, 154], [851, 110], [890, 65], [922, 42], [944, 40], [955, 42], [961, 51], [963, 63], [963, 77], [962, 100], [962, 115], [964, 129], [968, 136], [970, 137], [971, 137], [974, 132], [975, 124]]];

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
    guessAttemptNumber = 0;
    render();
}

render();

//Add user score number
function createScore() {
    const scores = document.createElement('div') as HTMLDivElement;
    const userscoreP = document.createElement('P') as HTMLParagraphElement;
    userscoreP.className = 'userScoreP';
    scores.appendChild(userscoreP);
    userscoreP.innerText = 'User score now: 0';

    const aiscoreP = document.createElement('P') as HTMLParagraphElement;
    aiscoreP.className = 'aiScoreP';
    scores.appendChild(aiscoreP);
    aiscoreP.innerText = 'AI score now: 0';

    document.body.appendChild(scores);
}

function updateUserScore(value :number) {    
    const userscoreP = d3.select('.userScoreP');
    userscoreP.text("User score now: " + value.toString());
    userscore = value;
}

function updateAIScore(value :number) {    
    const aiscoreP = d3.select('.aiscoreP');
    aiscoreP.text("AI score now: " + value.toString());
    aiscore = value;
}

createScore();

//Add reset button
const button = document.createElement('button');
button.innerHTML = 'reset';
button.addEventListener('click', reset as any);
document.body.appendChild(button);

socket.addEventListener('message', function(event) {
    const parsedData = JSON.parse(event.data);
    const eventType = parsedData.type;

    if (eventType === 'potential') {
        potential = [parsedData.data];
        render();
    } else if (eventType === 'ai_score') {
        updateAIScore(parsedData.score);
        aiguess = [parsedData.points];
        render();
    } else if (eventType === 'user_score') {
        updateUserScore(parsedData.score);
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

function sendResetMessage(level: number=0) {
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
