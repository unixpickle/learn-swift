import Vapor

let app = try await Application.make(.detect())

struct ListItem {
    var id: UInt64
    var text: String
}

actor TodoList {
    static let Global = TodoList()

    private var curId: UInt64 = 0
    private var list: [ListItem] = []

    func remove(id: UInt64) async -> Bool {
        if let idx = list.firstIndex(where: {x in x.id == id}) {
            list.remove(at: idx)
            return true
        }
        return false
    }

    func add(text: String) async {
        let id = curId
        curId += 1
        list.append(ListItem(id: id, text: text))
    }

    func items() async -> [ListItem] {
        // Should leverage copy-on-write automatically.
        return list
    }
}

app.get("") { req in
    var result = """
    <!doctype html>
    <html>
        <head>
            <title>Todo list</title>
        </head>
        <body>
    """
    let items = await TodoList.Global.items()
    if items.count == 0 {
        result += "<div>No items in list yet</div>"
    } else {
        result += "<table>"
        for item in items {
            result += "<tr><td>\(item.text)</td>"
            result += "<td><a href=\"/delete/\(item.id)\">Delete</a></td></tr>"
        }
        result += "</table>"
    }
    result += """
            <form action="/add" method="POST">
                Add item: <input name="text" id="addText">
                <input type="submit" value="Add">
            </form>
            <script>window.addText.focus()</script>
        </body>
    </html>
    """
    var headers = HTTPHeaders()
    headers.add(name: .contentType, value: "text/html")
    return Response(status: .ok, headers: headers, body: .init(string: result))
}

app.get("delete", ":id") { req in
    if let idStr = req.parameters.get("id"), let id = UInt64(idStr) {
        if await TodoList.Global.remove(id: id) {
            throw Abort.redirect(to: "/")
        }
        return Response(status: .ok, body: "item was not found")
    } else {
        return Response(status: .ok, body: "an invalid id was passed")
    }
}

struct AddBody: Content {
    var text: String
}

app.post("add") { req in
    let body = try req.content.decode(AddBody.self)
    await TodoList.Global.add(text: body.text)
    return req.redirect(to: "/")
}

try await app.execute()