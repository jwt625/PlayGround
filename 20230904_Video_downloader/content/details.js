(() => {
    "use strict";
    var e = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (e) {
            return typeof e
        } : function (e) {
            return e && "function" == typeof Symbol && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e
        },
        t = function () {
            function e(e, t) {
                for (var r = 0; r < t.length; r++) {
                    var n = t[r];
                    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, n.key, n)
                }
            }
            return function (t, r, n) {
                return r && e(t.prototype, r), n && e(t, n), t
            }
        }();

    function r(e, t) {
        if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
    }

    function n(e, t) {
        if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
        return !t || "object" != typeof t && "function" != typeof t ? e : t
    }

    function a(e, t) {
        if ("function" != typeof t && null !== t) throw new TypeError("Super expression must either be null or a function, not " + typeof t);
        e.prototype = Object.create(t && t.prototype, {
            constructor: {
                value: e,
                enumerable: !1,
                writable: !0,
                configurable: !0
            }
        }), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : e.__proto__ = t)
    }
    var o = createStore((function () {
            var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {
                    hit: null
                },
                t = arguments[1];
            return "setHit" == t.type ? e = {
                hit: t.payload
            } : "setError" == t.type && (e = {
                error: t.payload
            }), e
        })),
        l = decodeURIComponent(new URL(document.URL).hash.substr(1));
    weh.rpc.call("getHit", l).then((function (e) {
        e ? o.dispatch({
            type: "setHit",
            payload: e
        }) : o.dispatch({
            type: "setError",
            payload: weh._("no_such_hit")
        })
    })).catch((function (e) {
        o.dispatch({
            type: "setError",
            payload: e.message
        })
    }));
    var c = function (o) {
            function l() {
                return r(this, l), n(this, (l.__proto__ || Object.getPrototypeOf(l)).apply(this, arguments))
            }
            return a(l, React.Component), t(l, [{
                key: "renderValue",
                value: function () {
                    return "thumbnailUrl" == this.props.name || "thumbnail" == this.props.name ? React.createElement("div", null, React.createElement("img", {
                        src: this.props.value
                    }), React.createElement("br", null), React.createElement("div", {
                        className: "details-value"
                    }, this.props.value)) : null === this.props.value ? React.createElement("div", null, React.createElement("em", null, "null")) : "object" == e(this.props.value) ? React.createElement(ReactJson, {
                        src: this.props.value,
                        name: null,
                        collapsed: !0,
                        enableClipboard: !1,
                        collapseStringsAfterLength: 64,
                        displayDataTypes: !1,
                        displayObjectSize: !1,
                        style: {
                            display: "inline-block"
                        }
                    }) : React.createElement("div", {
                        className: "details-value"
                    }, "" + this.props.value)
                }
            }, {
                key: "render",
                value: function () {
                    return React.createElement("tr", null, React.createElement("td", null, this.props.name), React.createElement("td", null, this.renderValue()))
                }
            }]), l
        }(),
        i = connect((function (e, t) {
            return {
                hit: e.hit,
                error: e.error
            }
        }))(function (e) {
            function o() {
                return r(this, o), n(this, (o.__proto__ || Object.getPrototypeOf(o)).apply(this, arguments))
            }
            return a(o, React.Component), t(o, [{
                key: "renderError",
                value: function () {
                    return React.createElement("div", {
                        className: "details"
                    }, this.props.error)
                }
            }, {
                key: "render",
                value: function () {
                    if (this.props.error) return this.renderError();
                    if (!this.props.hit) return null;
                    var e = this,
                        t = Object.keys(this.props.hit).sort().map((function (t) {
                            return React.createElement(c, {
                                key: t,
                                name: t,
                                value: e.props.hit[t]
                            })
                        }));
                    return React.createElement("table", {
                        className: "details"
                    }, React.createElement("tbody", null, t))
                }
            }]), o
        }());
    render(React.createElement(Provider, {
        store: o
    }, React.createElement("div", {
        className: "weh-shf"
    }, React.createElement("div", null, React.createElement(WehHeader, {
        title: weh._("hit_details")
    }), React.createElement("main", null, React.createElement(i, null))))), document.getElementById("root")), weh.setPageTitle(weh._("hit_details"))
})();