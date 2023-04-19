import { Experiment } from "./experiment.js"

var allExperiments: Array<Experiment>

function createElement(type: string, props = {}, withText = ""): HTMLElement {
    var elem = document.createElement(type)
    for (const prop in props) {
        elem.setAttribute(prop, props[prop])
    }
    if (withText) {
        elem.textContent = withText
    }
    return elem
}

function renderExperiments() {
    const expsElem = document.getElementById("experiments")!
    for (const child of Array.from(expsElem.children)) {
        if (!child.className.includes("header")) {
            expsElem.removeChild(child)
        }
    }

    for (const exp of allExperiments) {
        const expElem = createElement("span", {class: "experiment"})
        expElem.appendChild(createElement("span", {class: "shortcode"}, exp.shortcode))
        expElem.appendChild(createElement("span", {class: "nepochs"}, exp.nepochs.toString()))
        expElem.appendChild(createElement("span", {class: "net_class"}, exp.net_class))
        expsElem.appendChild(expElem)
    }

}

async function loadExperiments() {
    var resp = await fetch("/nn-api/experiments")

    const data = await resp.text()
    if (resp.ok) {
        allExperiments = new Array()
        const expsIn = JSON.parse(data)
        for (const expIn of expsIn) {
            const exp = Experiment.from_json(expIn)
            allExperiments.push(exp)
        }
    }
    else {
        console.log(resp)
    }
}

loadExperiments().then((_val) => {
    console.log("fetched experiments")
    renderExperiments()
})