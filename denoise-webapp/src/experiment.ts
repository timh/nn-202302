class Experiment extends Object {
    shortcode: string
    net: any
    net_class: string = ""
    nepochs: number = 0

    constructor(shortcode: string) {
        super()
        this.shortcode = shortcode
    }

    static from_json(input: any): Experiment {
        const res = new Experiment(input.shortcode)
        res.net = new Object()
        res.net_class = input.net_args['class']
        res.nepochs = input.nepochs

        for (const key in input) {
            res.net[key] = input[key]
        }

        return res
    }
}

export { Experiment }