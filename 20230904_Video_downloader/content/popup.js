(() => {
    var e = {
            8: e => {
                e.exports = {
                    buildDate: "Tue Aug 22 2023 18:17:53 GMT+0200 (Central European Summer Time)",
                    buildOptions: {
                        browser: "firefox"
                    },
                    prod: !0
                }
            }
        },
        t = {};

    function a(n) {
        var i = t[n];
        if (void 0 !== i) return i.exports;
        var r = t[n] = {
            exports: {}
        };
        return e[n](r, r.exports, a), r.exports
    }(() => {
        "use strict";
        var e = function () {
            function e(e, t) {
                for (var a = 0; a < t.length; a++) {
                    var n = t[a];
                    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, n.key, n)
                }
            }
            return function (t, a, n) {
                return a && e(t.prototype, a), n && e(t, n), t
            }
        }();

        function t(e, t) {
            if (!(e instanceof t)) throw new TypeError("Cannot call a class as a function")
        }

        function n(e, t) {
            if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
            return !t || "object" != typeof t && "function" != typeof t ? e : t
        }

        function i(e, t) {
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
        var r = new RegExp("(\\d+)x(\\d+)"),
            o = new Set;
        window.addEventListener("unload", (function () {
            var e = !0,
                t = !1,
                a = void 0;
            try {
                for (var n, i = o[Symbol.iterator](); !(e = (n = i.next()).done); e = !0) {
                    var r = n.value;
                    weh.rpc.call("galleryUnhighlight", r)
                }
            } catch (e) {
                t = !0, a = e
            } finally {
                try {
                    !e && i.return && i.return()
                } finally {
                    if (t) throw a
                }
            }
        }));
        var s = a(8).buildOptions || {};
        "firefox" != s.browser && (document.querySelector("html").style.width = "500px");
        var c = createStore((function () {
            var e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {
                    hits: {},
                    progress: {},
                    actionHit: null,
                    actions: {},
                    logs: [],
                    embed: null,
                    maxHeight: void 0
                },
                t = arguments[1];
            switch (t.type) {
                case "setActionHit":
                    e = Object.assign({}, e, {
                        actionHit: t.payload
                    });
                    break;
                case "clearActionHit":
                    e = Object.assign({}, e, {
                        actionHit: null,
                        embed: null
                    });
                    break;
                case "updateData":
                    e = Object.assign({}, e, t.payload);
                    break;
                case "embed":
                    e = Object.assign({}, e, {
                        embed: t.payload
                    });
                    break;
                case "setMaxHeight":
                    e = Object.assign({}, e, {
                        maxHeight: t.payload
                    })
            }
            return e
        }));
        weh.rpc.listen({
            hits: function (e) {
                c.dispatch({
                    type: "updateData",
                    payload: {
                        hits: e
                    }
                })
            },
            progress: function (e) {
                c.dispatch({
                    type: "updateData",
                    payload: {
                        progress: e
                    }
                })
            },
            logs: function (e) {
                c.dispatch({
                    type: "updateData",
                    payload: {
                        logs: e
                    }
                })
            },
            copyToClipboard: function (e) {
                var t = document.createElement("input");
                document.body.appendChild(t), t.value = e, t.select(), document.execCommand("Copy"), document.body.removeChild(t)
            },
            embed: function (e) {
                c.dispatch({
                    type: "embed",
                    payload: e
                })
            }
        }), weh.prefs.on("popupHeightLeftOver", (function (e, t) {
            var a;
            switch (s.browser) {
                case "chrome":
                    a = 590;
                    break;
                case "edge":
                    a = 500;
                    break;
                default:
                    a = 600
            }
            browser.windows.getLastFocused().then((function (e) {
                c.dispatch({
                    type: "setMaxHeight",
                    payload: Math.min(a, e.height - t) + "px"
                })
            }))
        })), weh.rpc.call("getMainData").then((function (e) {
            c.dispatch({
                type: "updateData",
                payload: e
            })
        }));
        var l = connect((function (e, t) {
                return {
                    hits: e.hits || {},
                    progress: e.progress || {},
                    actionHit: e.actionHit || null,
                    actions: e.actions || {},
                    logs: e.logs || [],
                    embed: e.embed || null,
                    maxHeight: e.maxHeight || void 0
                }
            }), (function (e) {
                return bindActionCreators({}, e)
            }))(function (a) {
                function o(e) {
                    t(this, o);
                    var a = n(this, (o.__proto__ || Object.getPrototypeOf(o)).call(this, e));
                    return a.state = {
                        section: "active",
                        hitGroups: [],
                        hits: {},
                        counters: a.zeroCounters(),
                        actionHit: null
                    }, a.call = a.call.bind(a), a
                }
                return i(o, React.Component), e(o, [{
                    key: "componentWillReceiveProps",
                    value: function (e) {
                        this.buildGroups(e.hits), this.setState({
                            actionHit: e.actionHit
                        })
                    }
                }, {
                    key: "call",
                    value: function () {
                        for (var e = arguments.length, t = Array(e), a = 0; a < e; a++) t[a] = arguments[a];
                        return function () {
                            var e;
                            (e = weh.rpc).call.apply(e, t)
                        }
                    }
                }, {
                    key: "local",
                    value: function (e) {
                        for (var t, a = arguments.length, n = Array(a > 1 ? a - 1 : 0), i = 1; i < a; i++) n[i - 1] = arguments[i];
                        return (t = this[e]).bind.apply(t, [this].concat(n))
                    }
                }, {
                    key: "canClear",
                    value: function () {
                        for (var e = Object.keys(this.state.hits), t = 0; t < e.length; t++) {
                            var a = this.state.hits[e[t]];
                            if ("active" == a.status || "inactive" == a.status || "orphan" == a.status) return !0
                        }
                    }
                }, {
                    key: "zeroCounters",
                    value: function () {
                        return {
                            all: 0,
                            active: 0,
                            inactive: 0,
                            orphan: 0,
                            pinned: 0,
                            running: 0
                        }
                    }
                }, {
                    key: "buildGroups",
                    value: function (e, t) {
                        t = t || this.state.section;
                        var a = ["description", "descrPrefix", "adp", "size", "duration", "quality", "bitrate", "length", "mediaDomain", "type", "extension", "originalExt"],
                            n = {
                                counters: this.zeroCounters()
                            };
                        if (!e) return n.hitGroups = [], void this.setState(n);
                        var i = [],
                            o = {};
                        for (var s in e) {
                            var c = e[s];
                            c.status == t && (c.group ? o[c.group] ? o[c.group].push(c) : o[c.group] = [c] : (c.primary = !0, i.push([c]))), n.counters.all++, n.counters[c.status]++
                        }
                        for (var l in o) {
                            var u = o[l];
                            u.sort((function (e, t) {
                                if (e.bitrate && t.bitrate && e.bitrate != t.bitrate) return t.bitrate - e.bitrate;
                                var a = r.exec(e.size);
                                if (a) {
                                    var n = r.exec(t.size);
                                    if (n && (a[1] != n[1] || a[2] != n[2])) return parseInt(n[1]) * parseInt(n[2]) - parseInt(a[1]) * parseInt(a[2])
                                }
                                return 0
                            })), u.forEach((function (e, t) {
                                e.primary = 0 === t
                            }));
                            var h = u[0];
                            a.forEach((function (e) {
                                if (void 0 !== h[e]) {
                                    for (var t = !0, a = 1; a < u.length; a++) {
                                        if (u[a][e] !== h[e]) {
                                            t = !1;
                                            break
                                        }
                                    }
                                    if (t)
                                        for (a = 1; a < u.length; a++) delete u[a][e]
                                }
                            })), i.push(u)
                        }
                        n.hitGroups = i, n.hits = e, this.setState(n)
                    }
                }, {
                    key: "logDetails",
                    value: function (e) {
                        return function () {
                            weh.rpc.call("logDetails", e)
                        }
                    }
                }, {
                    key: "command",
                    value: function (e) {
                        var t = this;
                        return function (a) {
                            a.stopPropagation();
                            var n = a.shiftKey,
                                i = t.state.actionHit;
                            i && weh.rpc.call("actionCommand", e, i.id).then((function (e) {
                                n || e || window.close()
                            })), t.asDefaultInput && t.asDefaultInput.checked && (weh.prefs["default-action-" + (t.props.actions[e].catPriority || 0)] = e)
                        }
                    }
                }, {
                    key: "renderActions",
                    value: function () {
                        var e = this;
                        if (!this.state.actionHit || this.props.embed) return null;
                        var t = this,
                            a = (this.state.actionHit.actions || []).map((function (a) {
                                return React.createElement("div", {
                                    key: a,
                                    className: "vdh-container click action"
                                }, React.createElement("div", {
                                    onClick: e.command(a)
                                }, React.createElement("div", {
                                    className: "action-thumbnail"
                                }, React.createElement("img", {
                                    src: t.props.actions[a].icon
                                })), React.createElement("div", {
                                    className: "action-details"
                                }, React.createElement("div", {
                                    className: "action-title"
                                }, t.props.actions[a].title), React.createElement("div", {
                                    className: "action-descr"
                                }, t.props.actions[a].description))))
                            }));
                        return React.createElement("div", {
                            className: "actions"
                        }, a, React.createElement("div", {
                            className: "default-check"
                        }, React.createElement("input", {
                            id: "checkbox1",
                            type: "checkbox",
                            ref: function (t) {
                                return e.asDefaultInput = t
                            }
                        }), React.createElement("label", {
                            htmlFor: "checkbox1"
                        }, weh._("action_as_default"))))
                    }
                }, {
                    key: "renderEmbed",
                    value: function () {
                        return this.props.embed ? React.createElement(Embedded, {
                            className: "embed",
                            src: this.props.embed
                        }) : null
                    }
                }, {
                    key: "renderLog",
                    value: function () {
                        var e = this,
                            t = this.props.logs.map((function (t) {
                                return React.createElement("div", {
                                    key: t.key,
                                    className: "vdh-log vdh-log-" + t.type
                                }, t.videoTitle && React.createElement("div", {
                                    className: "log-video-title"
                                }, t.videoTitle), t.message, t.details && React.createElement("a", {
                                    onClick: e.logDetails(t.key),
                                    href: "#"
                                }, weh._("details_parenthesis")))
                            }));
                        return React.createElement("div", {
                            className: "logs"
                        }, t)
                    }
                }, {
                    key: "renderNoHit",
                    value: function () {
                        return "active" == this.state.section ? React.createElement("div", {
                            className: "no-media"
                        }, React.createElement("h2", null, weh._("no_media_current_tab")), React.createElement("p", null, weh._("no_media_to_process_descr"))) : React.createElement("div", {
                            className: "no-media"
                        }, React.createElement("h2", null, weh._("no_media_to_process")))
                    }
                }, {
                    key: "renderHits",
                    value: function () {
                        if ("log" == this.state.section) return this.renderLog();
                        if (0 == this.state.hitGroups.length) return this.renderNoHit();
                        var e = this,
                            t = [];
                        return Object.keys(this.state.hitGroups).forEach((function (a) {
                            e.state.hitGroups[a].forEach((function (a) {
                                var n = e.props.actions[a.actions[0]];
                                t.push(React.createElement(u, {
                                    key: a.id,
                                    hit: a,
                                    progress: e.props.progress[a.id] || 0,
                                    defaultAction: n
                                }))
                            }))
                        })), React.createElement("div", {
                            className: "has-media"
                        }, t)
                    }
                }, {
                    key: "renderFooterButtons",
                    value: function () {
                        return React.createElement("div", {
                            className: "buttons"
                        }, React.createElement("div", {
                            className: "buttons-container"
                        }, React.createElement("div", {
                            className: "buttons-opener"
                        }, React.createElement("img", {
                            src: "images/icon-3dots-64.png"
                        })), React.createElement("button", {
                            onClick: this.call("openAbout"),
                            title: weh._("about")
                        }, React.createElement("img", {
                            src: "images/icon-about-64.png"
                        })), "chrome" != s.browser && React.createElement("button", {
                            onClick: this.call("openSites"),
                            title: weh._("supported_sites")
                        }, React.createElement("img", {
                            src: "images/icon-sites-list-64.png"
                        })), React.createElement("button", {
                            onClick: this.call("analyzePage"),
                            title: weh._("analyze_page")
                        }, React.createElement("img", {
                            src: "images/icon-photo-64.png"
                        })), React.createElement("button", {
                            onClick: this.call("convertLocal"),
                            title: weh._("convert_local_files")
                        }, React.createElement("img", {
                            src: "images/icon-action-convert-b-64.png"
                        })), React.createElement("button", {
                            onClick: this.call("mergeLocal"),
                            title: weh._("merge_local_files")
                        }, React.createElement("img", {
                            src: "images/icon-merger-64.png"
                        })), React.createElement("div", {
                            className: "separator"
                        }), React.createElement("div", {
                            className: "separator"
                        }, " "), this.canClear() && React.createElement("button", {
                            onClick: this.call("clearHits", "all"),
                            title: weh._("clear_hits")
                        }, React.createElement("img", {
                            src: "images/icon-action-delete-64.png"
                        }))))
                    }
                }, {
                    key: "setSection",
                    value: function (e) {
                        var t = this;
                        return function () {
                            t.buildGroups(t.state.hits, e), t.setState({
                                section: e
                            })
                        }
                    }
                }, {
                    key: "clearLogs",
                    value: function () {
                        var e = this;
                        return function () {
                            weh.rpc.call("clearLogs").then(e.setSection("active"))
                        }
                    }
                }, {
                    key: "shouldDisplayGroupText",
                    value: function (e) {
                        if (this.state.section == e) return !1;
                        if ("log" == e && 0 == this.props.logs.length) return !1;
                        for (var t = Object.keys(this.state.counters), a = 0; a < t.length; a++) {
                            var n = t[a],
                                i = this.state.counters[n];
                            if (n == e && 0 == i) return !1;
                            if ("all" != n && n != e && n != this.state.section && i > 0) return !1
                        }
                        return !("log" != e && this.props.logs.length > 0)
                    }
                }, {
                    key: "showGroupBadge",
                    value: function (e, t) {
                        var a = "",
                            n = 0,
                            i = e;
                        if (this.state.section == e) return null;
                        if ("log" == e) {
                            if (0 == (n = this.props.logs.length)) return null;
                            var r = this.props.logs.filter((function (e) {
                                return "error" == e.type
                            }));
                            r.length > 0 ? (a = weh._("errors"), n = r.length, e = "error") : a = weh._("logs")
                        } else {
                            if (0 == this.state.counters[e]) return;
                            a = weh._(t), n = this.state.counters[e]
                        }
                        return React.createElement("div", {
                            className: "click group group-" + e,
                            onClick: this.setSection(i),
                            title: a
                        }, React.createElement("div", null, React.createElement("div", null, n)))
                    }
                }, {
                    key: "showGroupText",
                    value: function (e, t) {
                        if (this.shouldDisplayGroupText(e)) {
                            var a = this.props.logs.filter((function (e) {
                                return "error" == e.type
                            })).length;
                            return "log" == e && a > 0 && (t = "errors"), React.createElement("div", {
                                className: "click group-text",
                                onClick: this.setSection(e)
                            }, weh._(t))
                        }
                        return null
                    }
                }, {
                    key: "renderFooterGroups",
                    value: function () {
                        return React.createElement("div", {
                            className: "groups"
                        }, this.showGroupBadge("active", "in_current_tab"), this.showGroupText("active", "in_current_tab"), this.showGroupBadge("inactive", "in_other_tab"), this.showGroupText("inactive", "in_other_tab"), this.showGroupBadge("orphan", "orphan"), this.showGroupText("orphan", "orphan"), this.showGroupBadge("pinned", "pinned"), this.showGroupText("pinned", "pinned"), this.showGroupBadge("running", "running"), this.showGroupText("running", "running"), this.showGroupBadge("log", "logs"), this.showGroupText("log", "logs"))
                    }
                }, {
                    key: "renderFooter",
                    value: function () {
                        return React.createElement("footer", null, React.createElement("div", {
                            className: "right-side"
                        }, React.createElement("div", {
                            className: "separator"
                        }, " "), React.createElement("button", {
                            onClick: this.call("openSettings"),
                            title: weh._("settings")
                        }, React.createElement("img", {
                            src: "images/icon-settings-64.png"
                        }))), this.renderFooterButtons(), this.renderFooterGroups())
                    }
                }, {
                    key: "clearActionHit",
                    value: function () {
                        return function () {
                            c.dispatch({
                                type: "clearActionHit"
                            })
                        }
                    }
                }, {
                    key: "render",
                    value: function () {
                        return React.createElement("div", {
                            className: "main-panel " + (this.state.actionHit || this.props.embed ? "actions-open " : " ")
                        }, this.renderActions(), this.renderEmbed(), React.createElement("div", {
                            className: "main-content section-" + this.state.section,
                            style: {
                                maxHeight: this.props.maxHeight
                            }
                        }, React.createElement("img", {
                            className: "click back-active",
                            onClick: this.setSection("active"),
                            src: "images/icon-chevron-left-64.png"
                        }), "log" == this.state.section && React.createElement("img", {
                            className: "click clear-logs",
                            onClick: this.clearLogs(),
                            src: "images/icon-action-delete-64.png",
                            title: weh._("clear_logs")
                        }), React.createElement("div", {
                            className: "click back-margin",
                            onClick: this.setSection("active")
                        }), React.createElement("main", {
                            className: "content-hits"
                        }, this.renderHits(), React.createElement("div", {
                            className: "main-content-mask",
                            onClick: this.clearActionHit()
                        })), this.renderFooter()))
                    }
                }]), o
            }()),
            u = function (a) {
                function r(e) {
                    t(this, r);
                    var a = n(this, (r.__proto__ || Object.getPrototypeOf(r)).call(this, e));
                    return a.action = a.action.bind(a), a.state = {}, a
                }
                return i(r, React.Component), e(r, [{
                    key: "componentWillReceiveProps",
                    value: function (e) {
                        this.orphanTimer || this.updateOrphanTimer(e)
                    }
                }, {
                    key: "componentDidMount",
                    value: function () {
                        this.orphanTimer || this.updateOrphanTimer()
                    }
                }, {
                    key: "componentWillUnmount",
                    value: function () {
                        this.orphanTimer && (clearTimeout(this.orphanTimer), this.orphanTimer = null)
                    }
                }, {
                    key: "updateOrphanTimer",
                    value: function (e) {
                        if (e = e || this.props, this.orphanTimer = null, "orphan" == this.props.hit.status) {
                            var t = Date.now(),
                                a = this.props.hit.orphanT0,
                                n = this.props.hit.orphanT;
                            this.setState({
                                orphanTimer: Math.max(0, Math.min(100, 100 * (n - t) / (n - a)))
                            }), this.orphanTimer = setTimeout(this.updateOrphanTimer.bind(this), 1e3)
                        }
                    }
                }, {
                    key: "action",
                    value: function (e) {}
                }, {
                    key: "getClass",
                    value: function () {
                        return ""
                    }
                }, {
                    key: "durationString",
                    value: function (e) {
                        var t = Math.floor(e / 3600),
                            a = Math.floor(e % 3600 / 60),
                            n = e % 60;
                        return t > 0 ? t + ":" + ("00" + a).substr(-2) + ":" + ("00" + n).substr(-2) : a + ":" + ("00" + n).substr(-2)
                    }
                }, {
                    key: "description",
                    value: function () {
                        var e = this.props.hit;
                        if (e.description) return e.description;
                        var t = [];
                        if (e.descrPrefix && t.push(e.descrPrefix), e.adp && t.push("ADP"), e.size && t.push(e.size), e.duration && t.push(this.durationString(e.duration)), e.quality) {
                            var a = weh._("quality_" + e.quality);
                            "" == a && (a = e.quality.toUpperCase()), t.push(a)
                        }
                        if (e.bitrate) {
                            var n = e.bitrate,
                                i = "bps";
                            e.bitrate > 1e7 ? (i = "Mbps", n = Math.round(e.bitrate / 1e6)) : e.bitrate > 1e6 ? (i = "Mbps", n = Math.round(e.bitrate / 1e5) / 10) : e.bitrate > 1e4 ? (i = "Kbps", n = Math.round(e.bitrate / 1e3)) : e.bitrate > 1e3 && (i = "Kbps", n = Math.round(e.bitrate / 100) / 10), t.push(n + i)
                        }
                        var r = this.lengthString();
                        if (r && t.push(r), e.mediaDomain && t.push(weh._("from_domain", [e.mediaDomain])), "audio" == e.type && t.push(weh._("audio_only")), e.extension && (e.originalExt && e.originalExt != e.extension && t.push(e.originalExt.toUpperCase() + ">" + e.extension.toUpperCase()), t.push(e.extension.toUpperCase())), "downloading" === e.operation || "aggregating" === e.operation || "converting" === e.operation || "collecting" === e.operation || this.props.progress && !e.operation) {
                            var o = this.props.progress;
                            if (e.opStartDate && "undefined" !== o && (t.push(o + "%"), o > 0)) {
                                var s = (Date.now() - e.opStartDate) / o * (100 - o);
                                s = Math.max(0, Math.floor(s / 1e3)), t.push(this.durationString(s))
                            }
                        }
                        return t.join(" - ")
                    }
                }, {
                    key: "lengthString",
                    value: function () {
                        var e = this.props.hit;
                        return e.length ? e.length > 1048576 ? weh._("MB", [Math.round(10 * e.length / 1048576) / 10]) : e.length > 1024 ? weh._("KB", [Math.round(10 * e.length / 1024) / 10]) : weh._("Bytes", [e.length]) : null
                    }
                }, {
                    key: "titleClass",
                    value: function () {
                        var e = ["hit-title-text"];
                        return e.push("hit-title-text-" + weh.prefs.titleMode), e.join(" ")
                    }
                }, {
                    key: "progress",
                    value: function () {
                        return {
                            width: (this.props.progress || 0) + "%"
                        }
                    }
                }, {
                    key: "orphanTimerStyle",
                    value: function () {
                        return {
                            width: (this.state.orphanTimer || 100) + "%"
                        }
                    }
                }, {
                    key: "moreActions",
                    value: function () {
                        var e = this;
                        return function (t) {
                            t.stopPropagation(), c.dispatch({
                                type: "setActionHit",
                                payload: e.props.hit
                            })
                        }
                    }
                }, {
                    key: "call",
                    value: function () {
                        for (var e = arguments.length, t = Array(e), a = 0; a < e; a++) t[a] = arguments[a];
                        return function () {
                            var e;
                            (e = weh.rpc).call.apply(e, t)
                        }
                    }
                }, {
                    key: "callDefault",
                    value: function () {
                        var e = this;
                        return function (t) {
                            t.stopPropagation();
                            var a = t.shiftKey;
                            weh.rpc.call("actionCommand", e.props.hit.actions[0], e.props.hit.id).then((function (e) {
                                a || e || window.close()
                            }))
                        }
                    }
                }, {
                    key: "onMouseEnter",
                    value: function () {
                        var e = this;
                        return function () {
                            o.add(e.props.hit.selectorAttr), weh.rpc.call("galleryHighlight", e.props.hit.selectorAttr)
                        }
                    }
                }, {
                    key: "onMouseLeave",
                    value: function () {
                        var e = this;
                        return function () {
                            o.delete(e.props.hit.selectorAttr), weh.rpc.call("galleryUnhighlight", e.props.hit.selectorAttr)
                        }
                    }
                }, {
                    key: "render",
                    value: function () {
                        var e = this.props.hit;
                        return React.createElement("div", {
                            className: "click hit " + this.getClass(),
                            onMouseEnter: e.mouseTrack && this.onMouseEnter(),
                            onMouseLeave: e.mouseTrack && this.onMouseLeave(),
                            onClick: this.callDefault()
                        }, React.createElement("div", {
                            className: "vdh-container"
                        }, React.createElement("div", null, (e.thumbnail || e.thumbnailUrl) && e.primary && React.createElement("div", {
                            className: "hit-thumbnail"
                        }, React.createElement("img", {
                            src: e.thumbnailUrl || e.thumbnail
                        })), React.createElement("div", {
                            className: "vdh-fullwidth hit-descr"
                        }, e.primary && React.createElement("div", {
                            className: "hit-title"
                        }, React.createElement("div", {
                            className: this.titleClass()
                        }, e.title)), React.createElement("div", {
                            className: "hit-summary"
                        }, React.createElement("div", {
                            className: "hit-summary-action",
                            title: this.props.defaultAction.description
                        }, React.createElement("div", null, React.createElement("img", {
                            className: "default-action",
                            src: this.props.defaultAction.icon
                        }))), e.primary && weh.prefs.hitsGotoTab && e.topUrl && "inactive" == e.status && React.createElement("img", {
                            className: "hit-descr-button",
                            src: "images/icon-gototab-64.png",
                            title: weh._("hit_go_to_tab"),
                            onClick: this.call("gotoTab", e.topUrl)
                        }), e._signature && e._signature && e._signature.length > 0 && React.createElement("img", {
                            src: "images/icon-lock-64.png"
                        }), e.operation && React.createElement("span", {
                            className: "hit-operation"
                        }, weh._(e.operation) + " - "), this.description()), "running" == e.status && React.createElement("div", {
                            className: "hit-progress"
                        }, React.createElement("div", {
                            style: this.progress()
                        })), "orphan" == e.status && React.createElement("div", {
                            className: "hit-progress hit-orphan"
                        }, React.createElement("div", {
                            style: this.orphanTimerStyle()
                        }))), React.createElement("div", {
                            className: "click hit-actions",
                            onClick: this.moreActions()
                        }, React.createElement("div", null, React.createElement("img", {
                            className: "more-actions",
                            src: "images/icon-3dots-64.png"
                        }))))))
                    }
                }]), r
            }();
        render(React.createElement(Provider, {
            store: c
        }, React.createElement(l, null)), document.getElementById("root"))
    })()
})();